"""Train the single-neuron (logistic regression) fusion layer on top of the
out-of-fold LBP/HOG/BSIF CRC scores already produced by train_mad.py, and
report the standard neural-network training diagnostics: train/test
accuracy, the binary cross-entropy loss curve, and the learned weights and
bias.

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

    model = LogisticNeuron(lr=0.1, epochs=2000, seed=0).fit(X_train, y_train)

    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)
    final_loss = model.loss_history_[-1]

    print(f"\nTraining accuracy: {train_acc * 100:.2f}%")
    print(f"Test accuracy:     {test_acc * 100:.2f}%")
    print(f"Final training cross-entropy loss: {final_loss:.4f}")
    print(f"Weights -- lbp: {model.weights[0]:.4f}  hog: {model.weights[1]:.4f}  bsif: {model.weights[2]:.4f}")
    print(f"Bias: {model.bias:.4f}")

    metrics = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_accuracy_percent": train_acc * 100,
        "test_accuracy_percent": test_acc * 100,
        "final_train_cross_entropy_loss": final_loss,
        "weights": {
            "lbp": float(model.weights[0]),
            "hog": float(model.weights[1]),
            "bsif": float(model.weights[2]),
        },
        "bias": float(model.bias),
        "epochs": model.epochs,
        "learning_rate": model.lr,
    }
    with open(os.path.join(OUT_DIR, "nn_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(model.loss_history_, color="#2c3e50", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy loss")
    ax.set_title("Fusion-neuron training loss")
    ax.grid(True, ls=":", lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "nn_loss_curve.png"), dpi=150)
    print("\nSaved outputs/mad/nn_metrics.json and outputs/mad/nn_loss_curve.png")


if __name__ == "__main__":
    main()
