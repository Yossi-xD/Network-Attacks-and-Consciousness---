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


def relu(z):
    return np.maximum(z, 0.0)


def _binary_cross_entropy(y, p):
    eps = 1e-9
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def _balanced_sample_weight(y):
    """sklearn-style 'balanced' weighting: w_i = n_samples / (n_classes * count[y_i])."""
    n = len(y)
    counts = {c: np.sum(y == c) for c in np.unique(y)}
    w = np.array([n / (len(counts) * counts[yi]) for yi in y])
    return w


def _clip_grad(grad_w, grad_b, max_norm):
    if max_norm is None:
        return grad_w, grad_b
    norm = np.sqrt(np.sum(grad_w ** 2) + grad_b ** 2)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        grad_w = grad_w * scale
        grad_b = grad_b * scale
    return grad_w, grad_b


class LogisticNeuron:
    """z = w.x + b, p = sigmoid(z), trained by mini-batch SGD on BCE loss.

    The extra keyword-only knobs (weight_decay, lr_decay, momentum,
    class_weight, grad_clip, label_smoothing) all default to their original
    off/no-op values, so existing callers (train_mad_nn.py) reproduce the
    original dashboard unchanged. They exist to let a training-diagnostics
    comparison turn each one on independently.
    """

    def __init__(self, lr=0.5, n_iterations=1000, batch_size=8, seed=0,
                 weight_decay=0.0, lr_decay=0.0, momentum=0.0,
                 class_weight=None, grad_clip=None, label_smoothing=0.0):
        self.lr = lr
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.seed = seed
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.class_weight = class_weight
        self.grad_clip = grad_clip
        self.label_smoothing = label_smoothing

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

        sample_w = _balanced_sample_weight(y) if self.class_weight == "balanced" else np.ones(n)
        y_smooth = y * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0, 0.01, size=d)
        self.bias = 0.0
        vel_w, vel_b = np.zeros(d), 0.0

        history = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": [],
                   "weights": [], "bias": []}

        batch_size = min(self.batch_size, n)
        for t in range(self.n_iterations):
            idx = rng.choice(n, size=batch_size, replace=False)
            xb, yb, wb = Xn[idx], y_smooth[idx], sample_w[idx]

            p_batch = sigmoid(xb @ self.weights + self.bias)
            grad_z = wb * (p_batch - yb) / batch_size
            grad_w = xb.T @ grad_z + self.weight_decay * self.weights
            grad_b = grad_z.sum()
            grad_w, grad_b = _clip_grad(grad_w, grad_b, self.grad_clip)

            if self.momentum > 0:
                vel_w = self.momentum * vel_w + grad_w
                vel_b = self.momentum * vel_b + grad_b
                step_w, step_b = vel_w, vel_b
            else:
                step_w, step_b = grad_w, grad_b

            lr_t = self.lr / (1.0 + self.lr_decay * t)
            self.weights -= lr_t * step_w
            self.bias -= lr_t * step_b

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


class TinyMLP:
    """3 -> hidden -> 1 nonlinear fusion head: ReLU hidden layer, sigmoid
    output, trained by mini-batch SGD on BCE loss with the same optional
    regularization/optimization knobs as LogisticNeuron, so it's a direct,
    like-for-like capacity upgrade over the single neuron rather than a
    different training recipe.
    """

    def __init__(self, hidden=8, lr=0.5, n_iterations=1000, batch_size=8, seed=0,
                 weight_decay=0.0, lr_decay=0.0, momentum=0.0,
                 class_weight=None, grad_clip=None, label_smoothing=0.0):
        self.hidden = hidden
        self.lr = lr
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.seed = seed
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.class_weight = class_weight
        self.grad_clip = grad_clip
        self.label_smoothing = label_smoothing

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        h = self.hidden

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-6
        Xn = (X - self.mean_) / self.std_

        sample_w = _balanced_sample_weight(y) if self.class_weight == "balanced" else np.ones(n)
        y_smooth = y * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0, 0.5, size=(d, h)) / np.sqrt(d)
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, 0.5, size=h) / np.sqrt(h)
        self.b2 = 0.0
        vel = {k: np.zeros_like(getattr(self, k)) for k in ("W1", "b1", "W2", "b2")}

        self.history_ = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": [],
                          "w1_norm": [], "w2_norm": [], "b2": []}
        batch_size = min(self.batch_size, n)

        def forward(Xb):
            z1 = Xb @ self.W1 + self.b1
            a1 = relu(z1)
            z2 = a1 @ self.W2 + self.b2
            return sigmoid(z2), a1, z1

        for t in range(self.n_iterations):
            idx = rng.choice(n, size=batch_size, replace=False)
            xb, yb, wb = Xn[idx], y_smooth[idx], sample_w[idx]

            p, a1, z1 = forward(xb)
            dz2 = wb * (p - yb) / batch_size
            gW2 = a1.T @ dz2 + self.weight_decay * self.W2
            gb2 = dz2.sum()
            da1 = np.outer(dz2, self.W2)
            dz1 = da1 * (z1 > 0)
            gW1 = xb.T @ dz1 + self.weight_decay * self.W1
            gb1 = dz1.sum(axis=0)

            flat_g = np.concatenate([gW1.ravel(), gb1, gW2.ravel(), [gb2]])
            if self.grad_clip is not None:
                norm = np.linalg.norm(flat_g)
                if norm > self.grad_clip:
                    scale = self.grad_clip / (norm + 1e-12)
                    gW1, gb1, gW2, gb2 = gW1 * scale, gb1 * scale, gW2 * scale, gb2 * scale

            grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}
            lr_t = self.lr / (1.0 + self.lr_decay * t)
            for k in vel:
                if self.momentum > 0:
                    vel[k] = self.momentum * vel[k] + grads[k]
                    step = vel[k]
                else:
                    step = grads[k]
                setattr(self, k, getattr(self, k) - lr_t * step)

            p_train, _, _ = forward(Xn)
            self.history_["train_loss"].append(_binary_cross_entropy(y, p_train))
            self.history_["train_acc"].append(float(np.mean((p_train >= 0.5) == y)))
            self.history_["w1_norm"].append(float(np.linalg.norm(self.W1)))
            self.history_["w2_norm"].append(float(np.linalg.norm(self.W2)))
            self.history_["b2"].append(float(self.b2))
            if X_val is not None:
                Xn_val = (np.asarray(X_val, dtype=np.float64) - self.mean_) / self.std_
                p_val, _, _ = forward(Xn_val)
                self.history_["test_loss"].append(_binary_cross_entropy(np.asarray(y_val), p_val))
                self.history_["test_acc"].append(float(np.mean((p_val >= 0.5) == np.asarray(y_val))))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xn = (X - self.mean_) / self.std_
        a1 = relu(Xn @ self.W1 + self.b1)
        return sigmoid(a1 @ self.W2 + self.b2)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
