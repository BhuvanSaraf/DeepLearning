"""
perceptron.py

Single layer perceptron implemented from scratch (no ML/NN libraries).
Supports 3 activation functions:
    logistic -> for classification (targets should be 0/1)
    tanh     -> for classification (targets should be -1/1)
    linear   -> for regression

Trained using batch gradient descent on mean squared error.

Group 06
"""

import numpy as np


class Perceptron:

    def __init__(self, n_inputs, activation='logistic', learning_rate=0.01,
                 n_epochs=200, random_state=None):
        self.n_inputs = n_inputs
        self.activation = activation
        self.lr = learning_rate
        self.n_epochs = n_epochs

        if random_state is not None:
            np.random.seed(random_state)

        # weights[0] is the bias, weights[1:] are the input weights
        self.weights = np.random.uniform(-0.5, 0.5, n_inputs + 1)

        self.error_history_ = []   # avg error per epoch, filled in during fit()

    def net_input(self, X):
        # w0 + w1*x1 + w2*x2 + ...
        return self.weights[0] + np.dot(X, self.weights[1:])

    def activate(self, z):
        if self.activation == 'logistic':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'linear':
            return z
        else:
            raise ValueError('activation must be logistic, tanh or linear')

    def activate_deriv(self, a):
        # derivative of the activation, written in terms of the output a
        if self.activation == 'logistic':
            return a * (1 - a)
        elif self.activation == 'tanh':
            return 1 - a ** 2
        else:  # linear
            return np.ones_like(a)

    def predict(self, X):
        z = self.net_input(X)
        a = self.activate(z)

        if self.activation == 'logistic':
            return np.where(a >= 0.5, 1, 0)
        elif self.activation == 'tanh':
            return np.where(a >= 0, 1, -1)
        else:
            return a   # regression, just return the raw output

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples = X.shape[0]

        self.error_history_ = []

        for epoch in range(self.n_epochs):
            z = self.net_input(X)
            a = self.activate(z)

            error = y - a                      # (n_samples,)
            deriv = self.activate_deriv(a)
            delta = error * deriv

            # gradient descent update (batch, averaged over all samples)
            dw = np.dot(X.T, delta) / n_samples
            db = np.mean(delta)

            self.weights[1:] += self.lr * dw
            self.weights[0] += self.lr * db

            avg_error = np.mean(error ** 2) / 2
            self.error_history_.append(avg_error)

        return self


if __name__ == '__main__':
    # quick sanity check with a small linearly separable dataset
    np.random.seed(1)
    X0 = np.random.normal(loc=[-2, -2], scale=0.5, size=(50, 2))
    X1 = np.random.normal(loc=[2, 2], scale=0.5, size=(50, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)

    p = Perceptron(n_inputs=2, activation='logistic', learning_rate=0.1,
                    n_epochs=300, random_state=1)
    p.fit(X, y)
    preds = p.predict(X)
    print('training accuracy:', np.mean(preds == y))
