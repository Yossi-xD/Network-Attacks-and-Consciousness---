"""Generates one training-diagnostics dashboard PNG per step of the
mad_nn_experiments.py chain, plus a master progress chart and a DET-curve
comparison, all saved to outputs/mad/steps/.

Every step is retrained on the same fixed 138/36 GroupShuffleSplit that
train_mad_nn.py's original nn_dashboard.png uses, purely so each step's
figure is visually comparable to the original dashboard. The metrics that
actually matter (and that decided which steps were kept) are the 5-fold CV
numbers computed in mad_nn_experiments.run_chain() -- this script only adds
the pictures.
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
from mad_nn import TinyMLP
from mad_nn_experiments import run_chain, load_data, OUT_DIR
from train_mad import det_curve, find_deer

STEP_DIR = os.path.join(OUT_DIR, "steps")
STREAM_COLORS = {"lbp": "#2980b9", "hog": "#27ae60", "bsif": "#8e44ad"}
STREAM_NAMES_3 = ("lbp", "hog", "bsif")
STREAM_NAMES_2 = ("lbp", "hog")


def split_for(X, y, groups, seed=0):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def _finish(ax, stop_iter, legend=True):
    if stop_iter:
        ax.axvline(stop_iter, color="black", ls="--", lw=1.2, label=f"chosen stop ({stop_iter})")
    ax.set_xlabel("Iteration")
    ax.grid(True, ls=":", lw=0.5)
    if legend:
        ax.legend(fontsize=8)


def plot_linear_step(history, title, path, stream_names, stop_iter=None):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    ax = axes[0, 0]
    ax.plot(history["train_acc"], color="tab:blue", lw=0.6, alpha=0.8, label="training accuracy")
    ax.plot(history["test_acc"], color="tab:red", lw=1.8, label="test accuracy")
    ax.set_title("Accuracy"); ax.set_ylim(0, 1.02)
    _finish(ax, stop_iter)

    ax = axes[0, 1]
    ax.plot(history["train_loss"], color="tab:blue", lw=0.6, alpha=0.8, label="training loss")
    ax.plot(history["test_loss"], color="tab:red", lw=1.8, label="test loss")
    ax.set_title("Cross entropy loss")
    _finish(ax, stop_iter)

    ax = axes[1, 0]
    W = np.array(history["weights"])
    for i, name in enumerate(stream_names):
        ax.plot(W[:, i], color=STREAM_COLORS.get(name), lw=1.3, label=f"w_{name}")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Weights")
    _finish(ax, stop_iter)

    ax = axes[1, 1]
    ax.plot(history["bias"], color="#c0392b", lw=1.3, label="bias")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Bias")
    _finish(ax, stop_iter)

    fig.suptitle(title, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_mlp_step(history, title, path, stop_iter=None):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    ax = axes[0, 0]
    ax.plot(history["train_acc"], color="tab:blue", lw=0.6, alpha=0.8, label="training accuracy")
    ax.plot(history["test_acc"], color="tab:red", lw=1.8, label="test accuracy")
    ax.set_title("Accuracy"); ax.set_ylim(0, 1.02)
    _finish(ax, stop_iter)

    ax = axes[0, 1]
    ax.plot(history["train_loss"], color="tab:blue", lw=0.6, alpha=0.8, label="training loss")
    ax.plot(history["test_loss"], color="tab:red", lw=1.8, label="test loss")
    ax.set_title("Cross entropy loss")
    _finish(ax, stop_iter)

    ax = axes[1, 0]
    ax.plot(history["w1_norm"], color="#2980b9", lw=1.3, label="||W1|| (input->hidden)")
    ax.plot(history["w2_norm"], color="#27ae60", lw=1.3, label="||W2|| (hidden->output)")
    ax.set_title("Weight norms")
    _finish(ax, stop_iter)

    ax = axes[1, 1]
    ax.plot(history["b2"], color="#c0392b", lw=1.3, label="output bias (b2)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Output bias")
    _finish(ax, stop_iter)

    fig.suptitle(title, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_progress_chart(rows, path):
    steps = list(range(len(rows)))
    labels = [re.search(r"Step (\d+)", r["name"]).group(1) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(steps, [r["bpcer5"] for r in rows], "o-", color="#c0392b", lw=2, ms=6,
            label="BPCER @ APCER5 (lower is better)")
    ax.plot(steps, [r["deer"] for r in rows], "o-", color="#8e44ad", lw=2, ms=6,
            label="D-EER (lower is better)")
    ax.plot(steps, [r["f1"] for r in rows], "o--", color="#2980b9", lw=1.6, ms=5,
            label="F1 @ threshold 0.5")
    ax.plot(steps, [r["acc"] for r in rows], "o--", color="#27ae60", lw=1.6, ms=5,
            label="Accuracy @ threshold 0.5")
    ax.set_xticks(steps)
    ax.set_xticklabels([f"Step {l}" for l in labels], fontsize=9)
    ax.set_ylabel("%")
    ax.set_title("Every metric, every step (5-fold CV, out-of-fold predictions)")
    ax.grid(True, ls=":", lw=0.5)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_det_comparison(oof_base, oof_final, y, path, final_label):
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    for oof, label, color in ((oof_base, "Step 0 baseline", "#7f8c8d"),
                               (oof_final, final_label, "#c0392b")):
        apcers, bpcers, _ = det_curve(oof, y)
        deer, _ = find_deer(apcers, bpcers)
        order = np.argsort(apcers)
        ax.plot(np.clip(apcers[order], 1e-3, 1) * 100, np.clip(bpcers[order], 1e-3, 1) * 100,
                color=color, lw=1.8, label=f"{label} (D-EER={deer*100:.1f}%)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.1, 100); ax.set_ylim(0.1, 100)
    ax.set_xlabel("APCER (%)"); ax.set_ylabel("BPCER (%)")
    ax.set_title("DET curve -- baseline vs. recommended", fontsize=10)
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.axvline(5, color="black", lw=0.6, ls=":")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    os.makedirs(STEP_DIR, exist_ok=True)
    rows, step_configs = run_chain(verbose=True)

    X3, X2, y, groups = load_data()
    Xtr3, ytr3, Xte3, yte3 = split_for(X3, y, groups, seed=0)
    Xtr2, ytr2, Xte2, yte2 = split_for(X2, y, groups, seed=0)

    for row, sc in zip(rows, step_configs):
        tag = "step_" + re.search(r"Step (\d+)", row["name"]).group(1)
        path = os.path.join(STEP_DIR, f"{tag}_dashboard.png")
        Xtr, ytr, Xte, yte = (Xtr2, ytr2, Xte2, yte2) if sc["n_streams"] == 2 else (Xtr3, ytr3, Xte3, yte3)

        kwargs = dict(sc["kwargs"])
        stop_iter = None
        if kwargs.get("n_iterations", 1000) < 1000:
            stop_iter = kwargs["n_iterations"]
            kwargs["n_iterations"] = 1000  # show the full run with the chosen stop marked

        model = sc["cls"](**kwargs)
        model.fit(Xtr, ytr, X_val=Xte, y_val=yte)

        title = (f"{row['name']}\n"
                 f"5-fold CV: D-EER={row['deer']:.2f}%  BPCER@5={row['bpcer5']:.2f}%  F1={row['f1']:.2f}%")
        if sc["cls"] is TinyMLP:
            plot_mlp_step(model.history_, title, path, stop_iter=stop_iter)
        else:
            names = STREAM_NAMES_2 if sc["n_streams"] == 2 else STREAM_NAMES_3
            plot_linear_step(model.history_, title, path, names, stop_iter=stop_iter)
        print(f"saved steps/{os.path.basename(path)}")

    plot_progress_chart(rows, os.path.join(STEP_DIR, "progress_chart.png"))
    print("saved progress_chart.png")

    plot_det_comparison(rows[0]["oof"], rows[7]["oof"], y,
                         os.path.join(STEP_DIR, "det_comparison.png"),
                         final_label="Step 8 (early stopping)")
    print("saved det_comparison.png (baseline vs step 8)")


if __name__ == "__main__":
    main()
