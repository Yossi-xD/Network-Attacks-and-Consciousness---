"""Collaborative-representation classifier (Zhang, Yang & Feng, ICCV 2011),
used as the per-descriptor-stream classifier ("P-CRC") in Venkatesh et al.'s
ensemble-of-features S-MAD pipeline (FUSION 2020), plus sum-rule score
fusion across streams.

Each class (bona fide / morph) is represented by a dictionary made of its
own training feature vectors. A probe vector is reconstructed as a
ridge-regularized linear combination of each class's dictionary; the class
whose dictionary reconstructs it with lower residual is the more likely
label. Because the dictionary lives in sample-space (n_features x n_class
-samples), this stays cheap even when n_features is in the tens of
thousands, which is what the LBP/HOG/BSIF ensemble produces here.
"""
import numpy as np


class CRCClassifier:
    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-6
        Xn = (X - self.mean_) / self.std_
        self.dicts_ = {}
        self.proj_ = {}
        for c in (0, 1):
            D = Xn[y == c].T  # n_features x n_class_samples
            gram_inv = np.linalg.inv(D.T @ D + self.lam * np.eye(D.shape[1]))
            self.dicts_[c] = D
            self.proj_[c] = gram_inv @ D.T  # n_class_samples x n_features
        return self

    def _residual(self, x, c):
        alpha = self.proj_[c] @ x
        recon = self.dicts_[c] @ alpha
        return np.linalg.norm(x - recon)

    def score(self, X):
        """Morphing score per probe: bona fide residual minus morph
        residual. Higher => reconstructs better as a morph => more
        morph-like."""
        X = np.asarray(X, dtype=np.float64)
        Xn = (X - self.mean_) / self.std_
        scores = np.empty(len(Xn))
        for i, x in enumerate(Xn):
            r_bonafide = self._residual(x, 0)
            r_morph = self._residual(x, 1)
            scores[i] = r_bonafide - r_morph
        return scores


def fuse_scores(*score_arrays):
    """Sum-rule score-level fusion across descriptor streams."""
    return np.sum(np.stack(score_arrays, axis=0), axis=0)
