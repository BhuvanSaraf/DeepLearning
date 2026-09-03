import numpy as np


def train_test_split_per_class(X, y, train_frac=0.7, random_state=None):

    if random_state is not None:
        np.random.seed(random_state)

    X = np.array(X)
    y = np.array(y)

    train_idx = []
    test_idx = []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)
        n_train = int(train_frac * len(idx))
        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def one_against_one_pairs(classes):
    classes = sorted(classes)
    pairs = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            pairs.append((classes[i], classes[j]))
    return pairs


def subset_for_pair(X, y, class_a, class_b):
    mask = (y == class_a) | (y == class_b)
    return X[mask], y[mask]


def aggregate_votes(vote_matrix, classes):

    n_samples = vote_matrix.shape[0]
    final_pred = np.zeros(n_samples, dtype=vote_matrix.dtype)

    for i in range(n_samples):
        votes = vote_matrix[i]
        counts = {}
        for c in classes:
            counts[c] = 0
        for v in votes:
            counts[v] += 1

        best_class = classes[0]
        best_count = -1
        for c in classes:
            if counts[c] > best_count:
                best_count = counts[c]
                best_class = c
        final_pred[i] = best_class

    return final_pred


if __name__ == '__main__':
    X = np.random.randn(30, 2)
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    Xtr, Xte, ytr, yte = train_test_split_per_class(X, y, 0.7, random_state=0)
    print('train sizes per class:', [np.sum(ytr == c) for c in [0, 1, 2]])
    print('test sizes per class:', [np.sum(yte == c) for c in [0, 1, 2]])
    print('pairs:', one_against_one_pairs([0, 1, 2]))
