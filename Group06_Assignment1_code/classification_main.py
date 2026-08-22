"""
classification_main.py

Runs the full classification pipeline for the assignment:
  - loads Dataset 1 (linearly separable) and Dataset 2 (nonlinearly separable)
  - splits each into train/test (70/30 per class)
  - trains one perceptron per class pair (one-against-one), for both
    the logistic and tanh activations
  - saves error-vs-epoch plots and decision region plots (pairwise + combined)
  - prints confusion matrix, accuracy, precision/recall/f1 (per class,
    macro avg, micro avg) on the test set

Group 06
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron
from data_utils import (train_test_split_per_class, one_against_one_pairs,
                         subset_for_pair, aggregate_votes)
from metrics import print_all_metrics
from plotting import plot_error_curve, plot_decision_region

OUT_DIR = "results/classification"
DATA_DIR = "data/Group06"


def load_dataset(name="dataset1"):
    if name == "dataset1":
        # LS_Group06: one file per class, each row is "x1 x2"
        base = os.path.join(DATA_DIR, "Classification", "LS_Group06")
        X_list = []
        y_list = []
        for cls, fname in enumerate(["Class1.txt", "Class2.txt", "Class3.txt"]):
            Xc = np.loadtxt(os.path.join(base, fname))
            X_list.append(Xc)
            y_list.append(np.full(Xc.shape[0], cls))
        X = np.vstack(X_list)
        y = np.concatenate(y_list)

    elif name == "dataset2":
        # NLS_Group06.txt: first line is a header describing the split
        # (first 500 rows = class1, next 500 = class2, last 1000 = class3)
        path = os.path.join(DATA_DIR, "Classification", "NLS_Group06.txt")
        X = np.loadtxt(path, skiprows=1)
        y = np.concatenate([np.full(500, 0), np.full(500, 1), np.full(1000, 2)])

    else:
        raise ValueError(f"unknown dataset name {name}")

    return X, y


def train_one_against_one(X_train, y_train, activation, lr=0.2, n_epochs=2000, random_state=0):
    # trains one perceptron for every pair of classes
    classes = sorted(np.unique(y_train))
    pairs = one_against_one_pairs(classes)
    classifiers = {}

    for (a, b) in pairs:
        Xp, yp = subset_for_pair(X_train, y_train, a, b)

        if activation == 'logistic':
            target = np.where(yp == a, 0, 1)
        else:  # tanh
            target = np.where(yp == a, -1, 1)

        clf = Perceptron(n_inputs=2, activation=activation, learning_rate=lr,
                          n_epochs=n_epochs, random_state=random_state)
        clf.fit(Xp, target)
        classifiers[(a, b)] = clf

    return classifiers, pairs


def predict_pair(clf, X, class_a, class_b, activation):
    # maps a pairwise classifier's raw 0/1 or -1/1 output back to the
    # original class labels
    raw = clf.predict(X)
    if activation == 'logistic':
        return np.where(raw == 0, class_a, class_b)
    else:  # tanh
        return np.where(raw == -1, class_a, class_b)


def predict_combined(classifiers, pairs, X, activation):
    # one-against-one: run every pairwise classifier and take a majority vote
    classes = sorted({c for pair in pairs for c in pair})
    vote_matrix = np.empty((X.shape[0], len(pairs)), dtype=int)

    for k, (a, b) in enumerate(pairs):
        clf = classifiers[(a, b)]
        vote_matrix[:, k] = predict_pair(clf, X, a, b, activation)

    return aggregate_votes(vote_matrix, classes)


def run_for_dataset(dataset_name, out_subdir):
    os.makedirs(out_subdir, exist_ok=True)
    X, y = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split_per_class(X, y, train_frac=0.7, random_state=0)
    classes = sorted(np.unique(y))

    for activation in ['logistic', 'tanh']:
        print(f"\n=== {dataset_name} | activation = {activation} ===")
        classifiers, pairs = train_one_against_one(X_train, y_train, activation=activation,
                                                     lr=0.2, n_epochs=2000, random_state=0)

        # error vs epoch, one subplot per pairwise classifier
        fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))
        if len(pairs) == 1:
            axes = [axes]
        for ax, (a, b) in zip(axes, pairs):
            plot_error_curve(classifiers[(a, b)].error_history_,
                              title=f"Class {a} vs {b} ({activation})", ax=ax)
        fig.tight_layout()
        fig.savefig(f"{out_subdir}/{dataset_name}_{activation}_error_curves.png", dpi=150)
        plt.close(fig)

        # decision region for each class pair
        for (a, b) in pairs:
            clf = classifiers[(a, b)]
            Xp, yp = subset_for_pair(X_train, y_train, a, b)

            def predict_fn(grid, clf=clf, a=a, b=b):
                return predict_pair(clf, grid, a, b, activation)

            fig = plot_decision_region(Xp, yp, predict_fn,
                                        title=f"{dataset_name} decision region: {a} vs {b} ({activation})",
                                        save_path=f"{out_subdir}/{dataset_name}_{activation}_region_{a}v{b}.png")
            plt.close(fig)

        # combined decision region (all classes, majority vote)
        def predict_fn_combined(grid):
            return predict_combined(classifiers, pairs, grid, activation)

        fig = plot_decision_region(X_train, y_train, predict_fn_combined,
                                    title=f"{dataset_name} combined decision region ({activation})",
                                    save_path=f"{out_subdir}/{dataset_name}_{activation}_region_combined.png")
        plt.close(fig)

        # metrics on the test set
        y_pred_test = predict_combined(classifiers, pairs, X_test, activation)
        print("Test set metrics:")
        print_all_metrics(y_test, y_pred_test, labels=classes)


def main():
    run_for_dataset('dataset1', f"{OUT_DIR}/dataset1")
    run_for_dataset('dataset2', f"{OUT_DIR}/dataset2")
    print("\nDone. Plots saved under:", OUT_DIR)


if __name__ == '__main__':
    main()
