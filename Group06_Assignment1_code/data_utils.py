"""
data_utils.py
-------------
Helper utilities for splitting data and generating one-against-one
class pairs for multiclass perceptron classification.
"""

import numpy as np
from itertools import combinations


def train_test_split_per_class(X, y, train_frac=0.7, random_state=None,
                                shuffle=True):
    """
    Split (X, y) into train/test sets such that the split is done
    *within each class* (stratified), matching the assignment's
    requirement of a 70/30 split per class.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    train_frac : float
        Fraction of each class's samples to place in the training set.
    random_state : int or None
    shuffle : bool
        Shuffle samples within each class before splitting.

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarrays
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    train_idx = []
    test_idx = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if shuffle:
            idx = idx.copy()
            rng.shuffle(idx)
        n_train = int(round(len(idx) * train_frac))
        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def one_against_one_pairs(classes):
    """
    Return the list of (classA, classB) pairs for one-against-one
    multiclass decomposition.

    Example: classes = [0, 1, 2] -> [(0,1), (0,2), (1,2)]
    """
    return list(combinations(sorted(classes), 2))


def subset_for_pair(X, y, class_a, class_b):
    """Return the rows of X, y belonging only to class_a or class_b."""
    mask = (y == class_a) | (y == class_b)
    return X[mask], y[mask]


def one_against_one_vote(binary_predictions, class_a, class_b):
    """
    Given an array of binary predictions (already mapped back to the
    original class_a / class_b labels) from ONE pairwise classifier,
    return them unchanged -- this helper exists mainly for clarity
    when accumulating votes across multiple pairwise classifiers.
    Use `aggregate_votes` to combine predictions across all pairs.
    """
    return binary_predictions


def aggregate_votes(vote_matrix, classes):
    """
    vote_matrix : ndarray, shape (n_samples, n_pairs)
        vote_matrix[i, k] = the class label predicted by the k-th
        pairwise classifier for sample i.
    classes : list of all class labels

    Returns
    -------
    final_pred : ndarray, shape (n_samples,)
        The majority-vote class for each sample. Ties are broken by
        picking the lowest-indexed class among the tied classes.
    """
    n_samples = vote_matrix.shape[0]
    final_pred = np.empty(n_samples, dtype=vote_matrix.dtype)
    for i in range(n_samples):
        votes = vote_matrix[i]
        counts = {c: 0 for c in classes}
        for v in votes:
            counts[v] += 1
        # pick class with max votes; break ties by class order
        best_class = max(classes, key=lambda c: (counts[c], -classes.index(c)))
        final_pred[i] = best_class
    return final_pred


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    Xtr, Xte, ytr, yte = train_test_split_per_class(X, y, 0.7, random_state=0)
    print("train sizes per class:", [np.sum(ytr == c) for c in [0, 1, 2]])
    print("test sizes per class:", [np.sum(yte == c) for c in [0, 1, 2]])
    print("pairs:", one_against_one_pairs([0, 1, 2]))
