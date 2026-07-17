"""Part 2: train and evaluate the S-MAD ensemble-of-features + P-CRC pipeline
(Venkatesh, Raghavendra, Raja & Busch, FUSION 2020) on our own bona
fide/morph dataset from Part 1.

Evaluation follows ISO/IEC 30107-3: APCER (attacks classified as bona fide),
BPCER (bona fide classified as attacks), and D-EER (the operating point
where the two are equal). Because our dataset is small (58 morphs / 116
bona fide images) compared to the paper's (thousands), we use pair-disjoint
GroupKFold cross-validation instead of one fixed split, so every image gets
exactly one out-of-fold score and no image pair straddles train/test.
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from img_io import imread
from mad_crc import CRCClassifier, fuse_scores
from mad_features import bsif_stream, hog_stream, lbp_stream, learn_bsif_filters, scale_space_images

OUT_DIR = os.path.join(os.getcwd(), "outputs", "mad")
N_FOLDS = 5
CRC_LAMBDA = 1.0
STREAMS = ("lbp", "hog", "bsif")


def load_dataset():
    pair_ids = sorted({os.path.basename(p).split("_")[0] for p in glob.glob("outputs/pairs/*_A.png")})
    items = []  # (path, label, pair_id) -- label 0 = bona fide, 1 = morph
    for pid in pair_ids:
        items.append((f"outputs/pairs/{pid}_A.png", 0, pid))
        items.append((f"outputs/pairs/{pid}_B.png", 0, pid))
        morph_path = f"outputs/morphs/{pid}_morph.png"
        if os.path.exists(morph_path):
            items.append((morph_path, 1, pid))
    return items


def apcer_bpcer(scores, labels, threshold):
    morph_scores = scores[labels == 1]
    bf_scores = scores[labels == 0]
    apcer = float(np.mean(morph_scores <= threshold))
    bpcer = float(np.mean(bf_scores > threshold))
    return apcer, bpcer


def det_curve(scores, labels):
    thresholds = np.sort(np.unique(scores))
    apcers, bpcers = [], []
    for t in thresholds:
        a, b = apcer_bpcer(scores, labels, t)
        apcers.append(a)
        bpcers.append(b)
    return np.array(apcers), np.array(bpcers), thresholds


def find_deer(apcers, bpcers):
    idx = int(np.argmin(np.abs(apcers - bpcers)))
    return (apcers[idx] + bpcers[idx]) / 2, idx


def bpcer_at_apcer(apcers, bpcers, target):
    mask = apcers <= target
    if not np.any(mask):
        return 1.0
    return float(np.min(bpcers[mask]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    items = load_dataset()
    paths = [it[0] for it in items]
    labels = np.array([it[1] for it in items])
    groups = np.array([it[2] for it in items])
    n_bf, n_morph, n_groups = int(np.sum(labels == 0)), int(np.sum(labels == 1)), len(set(groups))
    print(f"Loaded {len(items)} images: {n_bf} bona fide, {n_morph} morph, {n_groups} pair-groups.")

    print("Reading images and building color/scale-space representation...")
    imgs = [imread(p) for p in paths]
    scale_cache = [scale_space_images(im) for im in imgs]

    print("Extracting LBP/HOG streams (fold-independent)...")
    lbp_all = [lbp_stream(s) for s in scale_cache]
    hog_all = [hog_stream(s) for s in scale_cache]
    print(f"  LBP dim={lbp_all[0].shape[0]}  HOG dim={hog_all[0].shape[0]}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_scores = np.zeros(len(items))
    oof_stream_scores = {s: np.zeros(len(items)) for s in STREAMS}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(paths, labels, groups)):
        print(f"Fold {fold + 1}/{N_FOLDS}: train={len(train_idx)} test={len(test_idx)}")

        train_bf_subimages = []
        for i in train_idx:
            if labels[i] == 0:
                train_bf_subimages.extend(scale_cache[i])
        filters = learn_bsif_filters(train_bf_subimages, seed=fold)
        bsif_all = [bsif_stream(s, filters) for s in scale_cache]

        stream_feats = {"lbp": lbp_all, "hog": hog_all, "bsif": bsif_all}
        fold_scores = []
        for stream in STREAMS:
            feats = stream_feats[stream]
            X_train = np.stack([feats[i] for i in train_idx])
            X_test = np.stack([feats[i] for i in test_idx])
            clf = CRCClassifier(lam=CRC_LAMBDA).fit(X_train, labels[train_idx])
            # z-score each stream against its own training-score distribution
            # before summing, so no single descriptor's residual scale (LBP,
            # HOG and BSIF have very different feature dimensionality) can
            # dominate the sum-rule fusion.
            train_scores = clf.score(X_train)
            mu, sigma = train_scores.mean(), train_scores.std() + 1e-6
            s = (clf.score(X_test) - mu) / sigma
            for j, i in enumerate(test_idx):
                oof_stream_scores[stream][i] = s[j]
            fold_scores.append(s)

        fused = fuse_scores(*fold_scores)
        for j, i in enumerate(test_idx):
            oof_scores[i] = fused[j]

    # ---- Evaluation (ISO/IEC 30107-3) ----
    apcers, bpcers, thresholds = det_curve(oof_scores, labels)
    deer, deer_idx = find_deer(apcers, bpcers)
    bpcer5 = bpcer_at_apcer(apcers, bpcers, 0.05)
    bpcer10 = bpcer_at_apcer(apcers, bpcers, 0.10)

    print("\n=== Results (pair-disjoint 5-fold cross-validation, fused score) ===")
    print(f"D-EER: {deer * 100:.2f}%")
    print(f"BPCER @ APCER=5%:  {bpcer5 * 100:.2f}%")
    print(f"BPCER @ APCER=10%: {bpcer10 * 100:.2f}%")

    per_stream_deer = {}
    for stream in STREAMS:
        a, b, _ = det_curve(oof_stream_scores[stream], labels)
        d, _ = find_deer(a, b)
        per_stream_deer[stream] = d
        print(f"  [{stream} alone] D-EER: {d * 100:.2f}%")

    metrics = {
        "n_bona_fide": n_bf,
        "n_morph": n_morph,
        "n_pair_groups": n_groups,
        "n_folds": N_FOLDS,
        "d_eer_percent": deer * 100,
        "bpcer_at_apcer5_percent": bpcer5 * 100,
        "bpcer_at_apcer10_percent": bpcer10 * 100,
        "per_stream_d_eer_percent": {k: v * 100 for k, v in per_stream_deer.items()},
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.savez(os.path.join(OUT_DIR, "scores.npz"), scores=oof_scores, labels=labels,
             groups=groups, paths=np.array(paths))

    # ---- DET curve plot ----
    fig, ax = plt.subplots(figsize=(5, 5))
    order = np.argsort(apcers)
    ax.plot(np.clip(apcers[order], 1e-3, 1) * 100, np.clip(bpcers[order], 1e-3, 1) * 100,
            color="#c0392b", lw=1.8, label="Ensemble (LBP+HOG+BSIF, fused)")
    ax.plot([deer * 100], [deer * 100], "ko", ms=5, label=f"D-EER = {deer * 100:.1f}%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 100)
    ax.set_ylim(0.1, 100)
    ax.set_xlabel("APCER (%)")
    ax.set_ylabel("BPCER (%)")
    ax.set_title("DET curve -- S-MAD ensemble-of-features\n(pair-disjoint cross-validation)", fontsize=10)
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "det_curve.png"), dpi=150)
    print("\nSaved metrics/scores/plot to outputs/mad")


if __name__ == "__main__":
    main()
