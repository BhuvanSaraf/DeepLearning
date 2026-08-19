"""
metrics.py
----------
From-scratch evaluation metrics: confusion matrix, accuracy,
RMSE and %RMSE. No sklearn / metrics libraries used.
"""

import numpy as np


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix.

    Parameters
    ----------
    y_true, y_pred : array-like of int/str class labels
    labels : list, optional
        Ordering of the labels for the matrix rows/columns.
        If None, inferred as the sorted unique labels of y_true/y_pred.

    Returns
    -------
    cm : ndarray of shape (n_classes, n_classes)
        cm[i, j] = number of samples with true label labels[i]
        predicted as labels[j].
    labels : list
        The label ordering used.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm, labels


def accuracy(y_true, y_pred):
    """Overall classification accuracy = correct / total."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def percent_rmse(y_true, y_pred):
    """
    %RMSE = 100 * RMSE / (max(y_true) - min(y_true))
    (a common normalization; range-based).
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    r = rmse(y_true, y_pred)
    span = y_true.max() - y_true.min()
    if span == 0:
        return float("nan")
    return 100.0 * r / span


def print_confusion_matrix(cm, labels):
    header = "      " + "  ".join(f"P{lab}" for lab in labels)
    print(header)
    for i, lab in enumerate(labels):
        row = "  ".join(f"{v:4d}" for v in cm[i])
        print(f"T{lab}   {row}")


if __name__ == "__main__":
    yt = [0, 0, 1, 1, 2, 2, 2]
    yp = [0, 1, 1, 1, 2, 2, 0]
    cm, labs = confusion_matrix(yt, yp)
    print_confusion_matrix(cm, labs)
    print("accuracy:", accuracy(yt, yp))

    yt_r = [1.0, 2.0, 3.0, 4.0]
    yp_r = [1.1, 1.9, 3.2, 3.7]
    print("rmse:", rmse(yt_r, yp_r))
    print("%rmse:", percent_rmse(yt_r, yp_r))
