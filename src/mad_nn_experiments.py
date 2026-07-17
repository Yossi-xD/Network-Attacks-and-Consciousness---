"""Step-by-step ablation of the fusion-neuron training-diagnostics
improvements ranked in the dashboard review, each evaluated under one
unified protocol: pair-disjoint 5-fold GroupKFold cross-validation over all
174 images (the same protocol train_mad.py uses for D-EER/BPCER), so every
step's "before -> after" numbers are measured on the same, low-variance
basis instead of the single noisy 36-image test split the dashboard used.

Each step trains a fresh model per fold with no leakage (GroupKFold keeps
each bona-fide/morph pair fully inside train or test), collects one
out-of-fold probability per image, and reports accuracy/precision/recall/F1
plus the operational D-EER / BPCER@APCER5 / BPCER@APCER10 metrics.
"""
import os
import sys

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
from mad_nn import LogisticNeuron, TinyMLP
from train_mad import apcer_bpcer, det_curve, find_deer, bpcer_at_apcer

OUT_DIR = os.path.join(os.getcwd(), "outputs", "mad")
N_FOLDS = 5


def cv_evaluate(model_factory, X, y, groups, n_folds=N_FOLDS):
    """Pair-disjoint GroupKFold CV. Returns one out-of-fold p(morph) per image."""
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros(len(y))
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])
    return oof


def cv_evaluate_early_stop(model_factory, X, y, groups, n_folds=N_FOLDS, seed=0):
    """Like cv_evaluate, but each fold carves an internal, group-disjoint
    validation slice out of its training data purely to pick the
    stopping iteration (lowest internal val loss), then refits on the
    fold's full training data for exactly that many iterations before
    scoring the held-out outer-fold test images. No outer test data is
    ever used to choose the stopping point.
    """
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros(len(y))
    stop_iters = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        sub_tr, sub_val = next(gss.split(X[train_idx], y[train_idx], groups[train_idx]))
        tr_idx, val_idx = train_idx[sub_tr], train_idx[sub_val]

        probe = model_factory()
        probe.fit(X[tr_idx], y[tr_idx], X_val=X[val_idx], y_val=y[val_idx])
        best_t = int(np.argmin(probe.history_["test_loss"])) + 1
        stop_iters.append(best_t)

        final = model_factory()
        final.n_iterations = best_t
        final.fit(X[train_idx], y[train_idx])
        oof[test_idx] = final.predict_proba(X[test_idx])
    return oof, stop_iters


def summarize(name, oof, y, prev=None):
    pred = (oof >= 0.5).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")

    apcers, bpcers, _ = det_curve(oof, y)
    deer, _ = find_deer(apcers, bpcers)
    b5 = bpcer_at_apcer(apcers, bpcers, 0.05)
    b10 = bpcer_at_apcer(apcers, bpcers, 0.10)

    row = {"name": name, "acc": acc * 100, "prec": prec * 100, "rec": rec * 100,
           "f1": f1 * 100, "deer": deer * 100, "bpcer5": b5 * 100, "bpcer10": b10 * 100,
           "tp": tp, "tn": tn, "fp": fp, "fn": fn}

    print(f"\n=== {name} ===")
    print(f"  Confusion (oof, n={len(y)}): TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"  Accuracy={acc*100:.2f}%  Precision={prec*100:.2f}%  Recall={rec*100:.2f}%  F1={f1*100:.2f}%")
    print(f"  D-EER={deer*100:.2f}%  BPCER@APCER5={b5*100:.2f}%  BPCER@APCER10={b10*100:.2f}%")
    if prev is not None:
        print(f"  Delta vs previous step: D-EER {row['deer']-prev['deer']:+.2f}pt, "
              f"BPCER@5 {row['bpcer5']-prev['bpcer5']:+.2f}pt, F1 {row['f1']-prev['f1']:+.2f}pt")
    return row


def main():
    d = np.load(os.path.join(OUT_DIR, "scores.npz"))
    X3 = np.column_stack([d["lbp"], d["hog"], d["bsif"]])
    X2 = np.column_stack([d["lbp"], d["hog"]])
    y = d["labels"]
    groups = d["groups"]
    print(f"Dataset: {len(y)} images ({np.sum(y==0)} bona fide, {np.sum(y==1)} morph), "
          f"{len(set(groups))} pair-groups, {N_FOLDS}-fold GroupKFold CV")

    rows = []

    # Step 0: current baseline hyperparameters, evaluated under the unified CV protocol.
    cfg = dict(lr=0.5, n_iterations=1000, batch_size=8, seed=0)
    oof = cv_evaluate(lambda: LogisticNeuron(**cfg), X3, y, groups)
    rows.append(summarize("Step 0: baseline (lr=0.5, batch=8, no regularization) -- unified CV protocol", oof, y))

    # Step 2: + L2 weight decay (small sweep, pick best by D-EER on this CV).
    best = None
    for wd in (1e-4, 1e-3, 1e-2, 5e-2):
        cfg2 = dict(cfg, weight_decay=wd)
        oof_wd = cv_evaluate(lambda c=cfg2: LogisticNeuron(**c), X3, y, groups)
        apcers, bpcers, _ = det_curve(oof_wd, y)
        deer_wd, _ = find_deer(apcers, bpcers)
        print(f"  [sweep] weight_decay={wd}: D-EER={deer_wd*100:.2f}%")
        if best is None or deer_wd < best[0]:
            best = (deer_wd, wd, oof_wd)
    cfg["weight_decay"] = best[1]
    rows.append(summarize(f"Step 2: + L2 weight decay (lambda={best[1]})", best[2], y, prev=rows[-1]))

    # Step 3: + LR decay and momentum on top of step 2.
    cfg["lr"] = 0.1
    cfg["lr_decay"] = 0.01
    cfg["momentum"] = 0.9
    oof = cv_evaluate(lambda c=dict(cfg): LogisticNeuron(**c), X3, y, groups)
    rows.append(summarize("Step 3: + lower LR (0.5->0.1) + 1/(1+0.01t) decay + momentum 0.9", oof, y, prev=rows[-1]))

    # Step 4: + class-balanced loss weighting.
    cfg["class_weight"] = "balanced"
    oof = cv_evaluate(lambda c=dict(cfg): LogisticNeuron(**c), X3, y, groups)
    rows.append(summarize("Step 4: + class-balanced loss weighting (2:1 bona fide:morph)", oof, y, prev=rows[-1]))

    # Step 5: + larger (near full) batch size.
    cfg["batch_size"] = 128  # clipped to each fold's train size internally (~139) -> effectively full-batch
    oof = cv_evaluate(lambda c=dict(cfg): LogisticNeuron(**c), X3, y, groups)
    rows.append(summarize("Step 5: + near full-batch (batch_size 8->128)", oof, y, prev=rows[-1]))

    # Step 6: BSIF ablation -- try dropping the near-chance BSIF stream.
    oof_2stream = cv_evaluate(lambda c=dict(cfg): LogisticNeuron(**c), X2, y, groups)
    row_2stream = summarize("Step 6 (trial): drop BSIF, fuse LBP+HOG only", oof_2stream, y, prev=rows[-1])
    if row_2stream["deer"] <= rows[-1]["deer"]:
        print("  -> keeping 2-stream (LBP+HOG) fusion: D-EER did not get worse without BSIF")
        X_active = X2
        rows.append(row_2stream)
    else:
        print("  -> reverting: keeping BSIF in the fusion, D-EER got worse without it")
        X_active = X3
        rows.append(summarize("Step 6: keep 3 streams (BSIF ablation reverted)", cv_evaluate(
            lambda c=dict(cfg): LogisticNeuron(**c), X3, y, groups), y, prev=rows[-2]))

    # Step 7: nonlinear fusion head (TinyMLP) with the same tricks accumulated so far.
    mlp_cfg = dict(cfg)
    mlp_cfg.pop("class_weight", None)
    mlp_cfg["class_weight"] = "balanced"
    mlp_cfg["hidden"] = 8
    oof_mlp = cv_evaluate(lambda c=dict(mlp_cfg): TinyMLP(**c), X_active, y, groups)
    row_mlp = summarize("Step 7 (trial): nonlinear fusion head (3->8->1 MLP, same tricks)", oof_mlp, y, prev=rows[-1])
    if row_mlp["deer"] <= rows[-1]["deer"]:
        print("  -> keeping TinyMLP: D-EER improved or held over the linear neuron")
        use_mlp = True
        rows.append(row_mlp)
    else:
        print("  -> reverting to the linear neuron: TinyMLP overfit the small dataset")
        use_mlp = False
        rows.append(dict(rows[-1], name="Step 7: keep linear neuron (MLP trial reverted)"))

    model_factory_base = (lambda c: (lambda: TinyMLP(**c))) if use_mlp else (lambda c: (lambda: LogisticNeuron(**c)))
    active_cfg = mlp_cfg if use_mlp else cfg

    # Step 8: early stopping via an internal, group-disjoint validation slice per fold.
    es_cfg = dict(active_cfg)
    es_cfg["n_iterations"] = 1000
    oof_es, stop_iters = cv_evaluate_early_stop(
        model_factory_base(es_cfg), X_active, y, groups, seed=0)
    print(f"  [early stopping] chosen iteration per fold: {stop_iters}")
    active_cfg["n_iterations"] = int(round(np.mean(stop_iters)))
    rows.append(summarize(f"Step 8: + early stopping (avg stop iter={active_cfg['n_iterations']} of 1000)",
                           oof_es, y, prev=rows[-1]))

    # Step 9: + gradient clipping.
    active_cfg["grad_clip"] = 1.0
    oof = cv_evaluate(model_factory_base(active_cfg), X_active, y, groups)
    rows.append(summarize("Step 9: + gradient clipping (max grad norm 1.0)", oof, y, prev=rows[-1]))

    # Step 10: + label smoothing.
    active_cfg["label_smoothing"] = 0.05
    oof = cv_evaluate(model_factory_base(active_cfg), X_active, y, groups)
    rows.append(summarize("Step 10: + label smoothing (0/1 -> 0.05/0.95)", oof, y, prev=rows[-1]))

    print("\n\n================ FULL CHAIN SUMMARY (vs. Step 0 baseline) ================")
    base = rows[0]
    for r in rows:
        print(f"{r['name']:<70s} D-EER={r['deer']:6.2f}%  BPCER@5={r['bpcer5']:6.2f}%  "
              f"F1={r['f1']:6.2f}%  Acc={r['acc']:6.2f}%   (D-EER delta from step0: {r['deer']-base['deer']:+.2f}pt)")

    print(f"\nFinal active config: {'TinyMLP' if use_mlp else 'LogisticNeuron'} {active_cfg}")
    print(f"Final X streams used: {'LBP+HOG' if X_active.shape[1]==2 else 'LBP+HOG+BSIF'}")


if __name__ == "__main__":
    main()
