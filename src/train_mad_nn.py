"""Train the single-neuron (logistic regression) fusion layer on top of the
out-of-fold LBP/HOG/BSIF CRC scores already produced by train_mad.py, and
report the standard neural-network training diagnostics dashboard:
train/test accuracy, train/test binary cross-entropy loss, and the weight
and bias trajectories, all vs. training iteration.

The 3 input features per image (one z-scored CRC residual score per
descriptor stream) are already leakage-free out-of-fold values from
train_mad.py's 5-fold pair-disjoint cross-validation, so this script only
needs one more pair-disjoint train/test split to fit and evaluate the
neuron itself.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
from mad_nn import LogisticNeuron

OUT_DIR = os.path.join(os.getcwd(), "outputs", "mad")
STREAMS = ("lbp", "hog", "bsif")
STREAM_COLORS = {"lbp": "#2980b9", "hog": "#27ae60", "bsif": "#8e44ad"}


def plot_dashboard(history, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    ax.plot(history["train_acc"], color="tab:blue", lw=0.6, alpha=0.8, label="training accuracy")
    ax.plot(history["test_acc"], color="tab:red", lw=2.0, label="test accuracy")
    ax.set_title("Accuracy")
    ax.set_xlabel("Iteration")
    ax.set_ylim(0, 1.02)
    ax.grid(True, ls=":", lw=0.5)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(history["train_loss"], color="tab:blue", lw=0.6, alpha=0.8, label="training loss")
    ax.plot(history["test_loss"], color="tab:red", lw=2.0, label="test loss")
    ax.set_title("Cross entropy loss")
    ax.set_xlabel("Iteration")
    ax.grid(True, ls=":", lw=0.5)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for i, stream in enumerate(STREAMS):
        ax.plot(history["weights"][:, i], color=STREAM_COLORS[stream], lw=1.3, label=f"w_{stream}")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Weights")
    ax.set_xlabel("Iteration")
    ax.grid(True, ls=":", lw=0.5)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(history["bias"], color="#c0392b", lw=1.3, label="bias")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Bias")
    ax.set_xlabel("Iteration")
    ax.grid(True, ls=":", lw=0.5)
    ax.legend(fontsize=8)

    fig.suptitle("Fusion-neuron training diagnostics (mini-batch SGD)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    scores_path = os.path.join(OUT_DIR, "scores.npz")
    d = np.load(scores_path)
    X = np.column_stack([d["lbp"], d["hog"], d["bsif"]])
    y = d["labels"]
    groups = d["groups"]

    n_bf, n_morph = int(np.sum(y == 0)), int(np.sum(y == 1))
    print(f"Fusion-neuron dataset: {len(X)} images ({n_bf} bona fide, {n_morph} morph)")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"Train: {len(train_idx)} images / {len(set(groups[train_idx]))} pairs  "
          f"Test: {len(test_idx)} images / {len(set(groups[test_idx]))} pairs")

    model = LogisticNeuron(lr=0.5, n_iterations=1000, batch_size=8, seed=0)
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    h = model.history_

    train_acc, test_acc = h["train_acc"][-1], h["test_acc"][-1]
    train_loss, test_loss = h["train_loss"][-1], h["test_loss"][-1]

    print(f"\nTraining accuracy: {train_acc * 100:.2f}%")
    print(f"Test accuracy:     {test_acc * 100:.2f}%")
    print(f"Final training cross-entropy loss: {train_loss:.4f}")
    print(f"Final test cross-entropy loss:     {test_loss:.4f}")
    print(f"Weights -- lbp: {model.weights[0]:.4f}  hog: {model.weights[1]:.4f}  bsif: {model.weights[2]:.4f}")
    print(f"Bias: {model.bias:.4f}")

    metrics = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_accuracy_percent": train_acc * 100,
        "test_accuracy_percent": test_acc * 100,
        "final_train_cross_entropy_loss": train_loss,
        "final_test_cross_entropy_loss": test_loss,
        "weights": {
            "lbp": float(model.weights[0]),
            "hog": float(model.weights[1]),
            "bsif": float(model.weights[2]),
        },
        "bias": float(model.bias),
        "n_iterations": model.n_iterations,
        "batch_size": model.batch_size,
        "learning_rate": model.lr,
    }
    with open(os.path.join(OUT_DIR, "nn_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_dashboard(h, os.path.join(OUT_DIR, "nn_dashboard.png"))
    print("\nSaved outputs/mad/nn_metrics.json and outputs/mad/nn_dashboard.png")


if __name__ == "__main__":
    main()
