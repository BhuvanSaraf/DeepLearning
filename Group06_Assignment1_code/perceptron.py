"""
perceptron.py
-------------
A single-layer perceptron implemented completely from scratch
(no ML/NN/gradient-descent libraries -- only plain Python + NumPy
arrays for storage and vectorized arithmetic).

Supports three activation functions:
    - 'logistic'  : sigmoid, for binary classification
    - 'tanh'      : tan-hyperbolic, for binary classification
    - 'linear'    : identity, for regression

Learning rule: batch gradient descent on the mean-squared-error
(equivalently, for logistic units this is the same update form
used in the standard perceptron/Adaline gradient-descent rule).

Author: Group 06
"""

import numpy as np


def _logistic(z):
    # numerically stable sigmoid
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _logistic_deriv(a):
    # derivative w.r.t. net input, expressed in terms of activation output a
    return a * (1.0 - a)


def _tanh(z):
    return np.tanh(z)


def _tanh_deriv(a):
    return 1.0 - a ** 2


def _linear(z):
    return z


def _linear_deriv(a):
    return np.ones_like(a)


_ACTIVATIONS = {
    "logistic": (_logistic, _logistic_deriv),
    "tanh": (_tanh, _tanh_deriv),
    "linear": (_linear, _linear_deriv),
}


class Perceptron:
    """
    Single-layer perceptron trained with batch gradient descent.

    Parameters
    ----------
    n_inputs : int
        Number of input features (bias is handled internally).
    activation : str
        One of 'logistic', 'tanh', 'linear'.
    learning_rate : float
        Gradient-descent step size (eta).
    n_epochs : int
        Maximum number of training epochs.
    tol : float
        Stop early if the change in average error between
        consecutive epochs falls below this value.
    random_state : int or None
        Seed for weight initialization, for reproducibility.

    Notes
    -----
    Targets for 'logistic' activation are expected in {0, 1}.
    Targets for 'tanh' activation are expected in {-1, 1}.
    Targets for 'linear' activation are the raw regression values.
    """

    def __init__(self, n_inputs, activation="logistic",
                 learning_rate=0.01, n_epochs=200, tol=1e-6,
                 random_state=None):
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. "
                              f"Choose from {list(_ACTIVATIONS)}.")
        self.n_inputs = n_inputs
        self.activation_name = activation
        self.act_fn, self.act_deriv = _ACTIVATIONS[activation]
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.tol = tol

        rng = np.random.default_rng(random_state)
        # weights[0] is the bias weight; weights[1:] correspond to inputs
        self.weights = rng.normal(loc=0.0, scale=0.01, size=n_inputs + 1)

        # populated after fit()
        self.error_history_ = []

    # ------------------------------------------------------------------
    def _augment(self, X):
        """Prepend a column of 1's for the bias term."""
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def net_input(self, X):
        Xa = self._augment(X)
        return Xa @ self.weights

    def predict_raw(self, X):
        """Return the raw activation output (before any thresholding)."""
        z = self.net_input(X)
        return self.act_fn(z)

    def predict(self, X):
        """
        Return class predictions for classification activations,
        or raw output for 'linear' (regression) activation.
        """
        out = self.predict_raw(X)
        if self.activation_name == "logistic":
            return (out >= 0.5).astype(int)
        elif self.activation_name == "tanh":
            return np.where(out >= 0.0, 1, -1)
        else:  # linear / regression
            return out

    # ------------------------------------------------------------------
    def fit(self, X, y, verbose=False):
        """
        Train using full-batch gradient descent on mean-squared error:
            E = (1/N) * sum( (t - a)^2 ) / 2

        Weight update (batch):
            dE/dw = -(1/N) * sum( (t - a) * f'(a) * x_augmented )
            w <- w - lr * dE/dw   (i.e. w <- w + lr * (1/N) * sum(...))
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        Xa = self._augment(X)
        n = X.shape[0]

        self.error_history_ = []
        prev_avg_err = None

        for epoch in range(self.n_epochs):
            z = Xa @ self.weights
            a = self.act_fn(z)
            error = y - a                      # (n,)
            deriv = self.act_deriv(a)           # (n,)
            delta = error * deriv               # (n,)

            grad = -(Xa * delta[:, None]).mean(axis=0)  # (n_inputs+1,)
            self.weights -= self.lr * grad

            # average error for this epoch (mean squared error / 2)
            avg_err = 0.5 * np.mean(error ** 2)
            self.error_history_.append(avg_err)

            if verbose and (epoch % max(1, self.n_epochs // 10) == 0):
                print(f"epoch {epoch:4d}  avg_error = {avg_err:.6f}")

            if prev_avg_err is not None and abs(prev_avg_err - avg_err) < self.tol:
                break
            prev_avg_err = avg_err

        return self

    # ------------------------------------------------------------------
    def get_weights(self):
        """Return (bias, weight_vector)."""
        return self.weights[0], self.weights[1:]


if __name__ == "__main__":
    # quick self-test with synthetic linearly separable data
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=[-2, -2], scale=0.5, size=(50, 2))
    X1 = rng.normal(loc=[2, 2], scale=0.5, size=(50, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)

    p = Perceptron(n_inputs=2, activation="logistic",
                    learning_rate=0.1, n_epochs=300, random_state=1)
    p.fit(X, y, verbose=True)
    preds = p.predict(X)
    acc = np.mean(preds == y)
    print(f"self-test training accuracy: {acc:.3f}")
