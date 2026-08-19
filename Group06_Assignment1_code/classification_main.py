"""
classification_main.py
-----------------------
End-to-end classification pipeline for the assignment:
  - Loads a dataset (2D features, integer class labels)
  - Splits 70/30 per class
  - Trains one-against-one perceptron classifiers for EACH class pair,
    for BOTH 'logistic' and 'tanh' activations
  - Produces: error-vs-epoch plots, pairwise + combined decision
    region plots, confusion matrix, accuracy

*** TODO once real data is provided ***
Replace `load_dataset()` with actual loading code for
Dataset 1 (linearly separable, 3 classes) and
Dataset 2 (nonlinearly separable, 2 or 3 classes).
Everything else in this file is dataset-agnostic and should not
need to change.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron
from data_utils import (train_test_split_per_class, one_against_one_pairs,
                         subset_for_pair, aggregate_votes)
from metrics import confusion_matrix, accuracy, print_confusion_matrix
from plotting import plot_error_curve, plot_decision_region

OUT_DIR = "results/classification"
DATA_DIR = "data/Group06"


# ----------------------------------------------------------------------
def load_dataset(name="dataset1"):
    """
    Real Group06 data loader.

    dataset1 -> LS_Group06: 3 linearly separable classes, 500 pts/class,
                one file per class (Class1.txt, Class2.txt, Class3.txt),
                each row is "x1 x2" (whitespace-separated, no header).

    dataset2 -> NLS_Group06.txt: 2000 points, 2D, nonlinearly separable.
                First line is a header comment describing the split:
                first 500 rows = class1, next 500 = class2,
                last 1000 = class3. We skip that header line and
                reconstruct labels accordingly.
    """
    if name == "dataset1":
        base = os.path.join(DATA_DIR, "Classification", "LS_Group06")
        X_list, y_list = [], []
        for cls, fname in enumerate(["Class1.txt", "Class2.txt", "Class3.txt"]):
            Xc = np.loadtxt(os.path.join(base, fname))
            X_list.append(Xc)
            y_list.append(np.full(Xc.shape[0], cls))
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
    elif name == "dataset2":
        path = os.path.join(DATA_DIR, "Classification", "NLS_Group06.txt")
        X = np.loadtxt(path, skiprows=1)  # skip the header/description line
        # Per the file's header: first 500 -> class0, next 500 -> class1,
        # last 1000 -> class2 (0-indexed here; file calls them class1/2/3)
        y = np.concatenate([np.full(500, 0), np.full(500, 1), np.full(1000, 2)])
    else:
        raise ValueError(f"unknown dataset name {name}")
    return X, y


# ----------------------------------------------------------------------
def train_one_against_one(X_train, y_train, activation, lr=0.05,
                           n_epochs=300, random_state=0):
    """
    Train one perceptron per class pair.

    Returns
    -------
    classifiers : dict {(class_a, class_b): Perceptron}
    """
    classes = sorted(np.unique(y_train))
    pairs = one_against_one_pairs(classes)
    classifiers = {}

    for (a, b) in pairs:
        Xp, yp = subset_for_pair(X_train, y_train, a, b)
        if activation == "logistic":
            target = np.where(yp == a, 0, 1)
        else:  # tanh
            target = np.where(yp == a, -1, 1)

        clf = Perceptron(n_inputs=2, activation=activation,
                          learning_rate=lr, n_epochs=n_epochs,
                          random_state=random_state)
        clf.fit(Xp, target)
        classifiers[(a, b)] = clf

    return classifiers, pairs


def predict_pair(clf, X, class_a, class_b, activation):
    """Map a pairwise classifier's raw prediction back to original labels."""
    raw = clf.predict(X)
    if activation == "logistic":
        return np.where(raw == 0, class_a, class_b)
    else:  # tanh
        return np.where(raw == -1, class_a, class_b)


def predict_combined(classifiers, pairs, X, activation):
    """One-against-one majority vote across all pairwise classifiers."""
    classes = sorted({c for pair in pairs for c in pair})
    vote_matrix = np.empty((X.shape[0], len(pairs)), dtype=int)
    for k, (a, b) in enumerate(pairs):
        clf = classifiers[(a, b)]
        vote_matrix[:, k] = predict_pair(clf, X, a, b, activation)
    return aggregate_votes(vote_matrix, classes)


# ----------------------------------------------------------------------
def run_for_dataset(dataset_name, out_subdir):
    os.makedirs(out_subdir, exist_ok=True)
    X, y = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split_per_class(
        X, y, train_frac=0.7, random_state=0)

    classes = sorted(np.unique(y))

    for activation in ["logistic", "tanh"]:
        print(f"\n=== {dataset_name} | activation = {activation} ===")
        classifiers, pairs = train_one_against_one(
            X_train, y_train, activation=activation,
            lr=0.05, n_epochs=300, random_state=0)

        # --- error vs epoch, one subplot per pairwise classifier ---
        fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))
        if len(pairs) == 1:
            axes = [axes]
        for ax, (a, b) in zip(axes, pairs):
            plot_error_curve(classifiers[(a, b)].error_history_,
                              title=f"Class {a} vs {b} ({activation})", ax=ax)
        fig.tight_layout()
        fig.savefig(f"{out_subdir}/{dataset_name}_{activation}_error_curves.png", dpi=150)
        plt.close(fig)

        # --- pairwise decision regions ---
        for (a, b) in pairs:
            clf = classifiers[(a, b)]
            Xp, yp = subset_for_pair(X_train, y_train, a, b)
            fn = lambda g, a=a, b=b, clf=clf: predict_pair(clf, g, a, b, activation)
            fig = plot_decision_region(
                Xp, yp, fn,
                title=f"{dataset_name} decision region: {a} vs {b} ({activation})",
                save_path=f"{out_subdir}/{dataset_name}_{activation}_region_{a}v{b}.png")
            plt.close(fig)

        # --- combined decision region ---
        fn_combined = lambda g: predict_combined(classifiers, pairs, g, activation)
        fig = plot_decision_region(
            X_train, y_train, fn_combined,
            title=f"{dataset_name} combined decision region ({activation})",
            save_path=f"{out_subdir}/{dataset_name}_{activation}_region_combined.png")
        plt.close(fig)

        # --- confusion matrix + accuracy on test data ---
        y_pred_test = predict_combined(classifiers, pairs, X_test, activation)
        cm, labs = confusion_matrix(y_test, y_pred_test, labels=classes)
        acc = accuracy(y_test, y_pred_test)
        print("Confusion matrix (test data):")
        print_confusion_matrix(cm, labs)
        print(f"Test accuracy: {acc:.4f}")


def main():
    run_for_dataset("dataset1", f"{OUT_DIR}/dataset1")
    run_for_dataset("dataset2", f"{OUT_DIR}/dataset2")
    print("\nDone. Plots saved under:", OUT_DIR)


if __name__ == "__main__":
    main()
