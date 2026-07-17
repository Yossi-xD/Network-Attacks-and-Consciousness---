"""The simplest possible neural network -- a single neuron (logistic
regression) -- used to learn the fusion weights across the LBP/HOG/BSIF
descriptor streams, replacing the hand-picked sum rule with a trained
weighted combination: z = w_lbp*lbp + w_hog*hog + w_bsif*bsif + b, p =
sigmoid(z), trained with full-batch gradient descent on the binary
cross-entropy loss.
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogisticNeuron:
    def __init__(self, lr=0.1, epochs=2000, seed=0):
        self.lr = lr
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-6
        Xn = (X - self.mean_) / self.std_

        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0, 0.01, size=d)
        self.bias = 0.0
        self.loss_history_ = []

        for _ in range(self.epochs):
            z = Xn @ self.weights + self.bias
            p = sigmoid(z)
            eps = 1e-9
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            self.loss_history_.append(float(loss))

            grad_z = (p - y) / n
            grad_w = Xn.T @ grad_z
            grad_b = grad_z.sum()
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xn = (X - self.mean_) / self.std_
        return sigmoid(Xn @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
