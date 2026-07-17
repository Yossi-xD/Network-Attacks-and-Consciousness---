"""The simplest possible neural network -- a single neuron (logistic
regression) -- used to learn the fusion weights across the LBP/HOG/BSIF
descriptor streams, replacing the hand-picked sum rule with a trained
weighted combination: z = w_lbp*lbp + w_hog*hog + w_bsif*bsif + b, p =
sigmoid(z), trained by mini-batch stochastic gradient descent on the binary
cross-entropy loss.

Mini-batches (rather than one full-batch update per step) are what give the
training accuracy/loss curves their characteristic step-to-step jitter,
while the per-iteration validation accuracy/loss and the full weight/bias
trajectory are tracked in `history_` for the training-diagnostics
dashboard.
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _binary_cross_entropy(y, p):
    eps = 1e-9
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


class LogisticNeuron:
    def __init__(self, lr=0.5, n_iterations=1000, batch_size=8, seed=0):
        self.lr = lr
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-6
        Xn = (X - self.mean_) / self.std_
        has_val = X_val is not None
        if has_val:
            Xn_val = (np.asarray(X_val, dtype=np.float64) - self.mean_) / self.std_
            y_val = np.asarray(y_val, dtype=np.float64)

        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0, 0.01, size=d)
        self.bias = 0.0

        history = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": [],
                   "weights": [], "bias": []}

        batch_size = min(self.batch_size, n)
        for _ in range(self.n_iterations):
            idx = rng.choice(n, size=batch_size, replace=False)
            xb, yb = Xn[idx], y[idx]

            p_batch = sigmoid(xb @ self.weights + self.bias)
            grad_z = (p_batch - yb) / batch_size
            self.weights -= self.lr * (xb.T @ grad_z)
            self.bias -= self.lr * grad_z.sum()

            p_train = sigmoid(Xn @ self.weights + self.bias)
            history["train_loss"].append(_binary_cross_entropy(y, p_train))
            history["train_acc"].append(float(np.mean((p_train >= 0.5) == y)))
            history["weights"].append(self.weights.copy())
            history["bias"].append(self.bias)

            if has_val:
                p_val = sigmoid(Xn_val @ self.weights + self.bias)
                history["test_loss"].append(_binary_cross_entropy(y_val, p_val))
                history["test_acc"].append(float(np.mean((p_val >= 0.5) == y_val)))

        history["weights"] = np.array(history["weights"])
        history["bias"] = np.array(history["bias"])
        self.history_ = history
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xn = (X - self.mean_) / self.std_
        return sigmoid(Xn @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
