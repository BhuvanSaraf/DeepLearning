"""
classification_main.py

Assignment 2 classification pipeline:
  - loads Dataset 1 (linearly separable) and Dataset 2 (nonlinearly separable)
  - splits each into train/val/test (60/20/20 per class)
  - trains an FCNN (from scratch, squared error loss, SGD) with
    1 hidden layer for Dataset 1 and 2 hidden layers for Dataset 2,
    trying several hidden-node counts and activations
  - prints full metrics (confusion matrix, per-class + mean precision/
    recall/F1) on the VALIDATION set for every architecture tried
  - picks the best architecture by validation accuracy, then reports
    error-vs-epoch, decision region, hidden/output node plots, and
    full test-set metrics for that architecture only
  - compares the best FCNN against the Assignment 1 single-neuron
    (perceptron) baseline on the same train/test split

Group 06
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork
from data_utils import (train_val_test_split_per_class, one_against_one_pairs,
                         subset_for_pair, aggregate_votes)
from metrics import print_all_metrics, accuracy
from plotting import plot_error_curve, plot_decision_region, plot_node_outputs
from perceptron_baseline import Perceptron

OUT_DIR = "results/classification"
DATA_DIR = "data/Group06"


def load_dataset(name="dataset1"):
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
        X = np.loadtxt(path, skiprows=1)
        y = np.concatenate([np.full(500, 0), np.full(500, 1), np.full(1000, 2)])

    else:
        raise ValueError(f"unknown dataset name {name}")

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def preprocess_features(X, mean, std):
    return (X - mean) / std


def one_hot_encode(y, classes):
    y = np.asarray(y, dtype=int)
    class_to_index = {cls: i for i, cls in enumerate(classes)}
    encoded = np.zeros((len(y), len(classes)), dtype=float)
    for i, label in enumerate(y):
        encoded[i, class_to_index[int(label)]] = 1.0
    return encoded


def predict_multiclass(model, X, classes):
    out = model.predict(X)
    pred_idx = np.argmax(out, axis=1)
    return np.asarray(classes)[pred_idx]


def train_fcnn(X_train, y_train_onehot, input_size, hidden_dims, n_classes,
                activation, lr, n_epochs, random_state=0):
    net_dims = [input_size] + list(hidden_dims) + [n_classes]
    model = NeuralNetwork(
        input_size=input_size,
        net_dims=net_dims,
        activation=activation,
        type='classification',
        seed=random_state,
        learning_rate=lr,
        weight_decay=0,
        batch_size=1,       # true SGD, one sample at a time
        epochs=n_epochs,
    )
    model.fit(X_train, y_train_onehot)
    return model


def run_for_dataset(dataset_name, hidden_options, lr, n_epochs, out_subdir, random_state=0):
    os.makedirs(out_subdir, exist_ok=True)

    X, y = load_dataset(dataset_name)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_per_class(
        X, y, train_frac=0.6, val_frac=0.2, random_state=random_state)

    # normalize using training statistics only
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train_n = preprocess_features(X_train, mean, std)
    X_val_n = preprocess_features(X_val, mean, std)
    X_test_n = preprocess_features(X_test, mean, std)

    classes = sorted(np.unique(y))
    y_train_oh = one_hot_encode(y_train, classes)

    print(f"\n{'=' * 70}")
    print(f"{dataset_name}: architecture search (hidden layers: {len(hidden_options[0])})")
    print(f"{'=' * 70}")

    search_results = []
    for hidden_dims in hidden_options:
        for activation in ['logistic', 'tanh']:
            model = train_fcnn(X_train_n, y_train_oh, X.shape[1], hidden_dims,
                                len(classes), activation, lr, n_epochs, random_state)
            y_pred_val = predict_multiclass(model, X_val_n, classes)

            print(f"\n--- hidden_dims={hidden_dims}, activation={activation} "
                  f"(validation set) ---")
            metrics = print_all_metrics(y_val, y_pred_val, labels=classes)

            search_results.append({
                'hidden_dims': hidden_dims,
                'activation': activation,
                'model': model,
                'val_accuracy': metrics['accuracy'],
                'val_macro_f1': metrics['macro_f1'],
            })

    # pick the best architecture by validation accuracy
    # (tie-break by macro F1)
    best = max(search_results, key=lambda r: (r['val_accuracy'], r['val_macro_f1']))
    print(f"\n>>> Best architecture for {dataset_name}: "
          f"hidden_dims={best['hidden_dims']}, activation={best['activation']} "
          f"(val accuracy={best['val_accuracy']:.4f})")

    best_model = best['model']
    activation = best['activation']

    # error vs epoch (best architecture only)
    fig = plot_error_curve(best_model.error_history_,
                            title=f"{dataset_name}: training error ({best['hidden_dims']}, {activation})",
                            save_path=f"{out_subdir}/{dataset_name}_best_error_curve.png")
    plt.close(fig)

    # decision region (best architecture only, training data superimposed)
    def predict_fn(grid, model=best_model, classes=classes):
        return predict_multiclass(model, grid, classes)

    fig = plot_decision_region(X_train_n, y_train, predict_fn,
                                title=f"{dataset_name}: decision region (best architecture)",
                                save_path=f"{out_subdir}/{dataset_name}_best_decision_region.png")
    plt.close(fig)

    # hidden / output node plots for train, val, test (colored by class)
    node_dir = f"{out_subdir}/{dataset_name}_node_outputs"
    for split_name, X_split, y_split in [('train', X_train_n, y_train),
                                           ('val', X_val_n, y_val),
                                           ('test', X_test_n, y_test)]:
        plot_node_outputs(best_model, X_split, input_dim=X.shape[1],
                           title_prefix=dataset_name, out_dir=node_dir,
                           split_name=split_name, y_labels=y_split)

    # full metrics on the test set for the best architecture
    y_pred_test = predict_multiclass(best_model, X_test_n, classes)
    print(f"\n{dataset_name}: TEST SET metrics for best architecture "
          f"(hidden_dims={best['hidden_dims']}, activation={activation})")
    print_all_metrics(y_test, y_pred_test, labels=classes)

    # ---- comparison against the Assignment 1 single-neuron perceptron ----
    print(f"\n{dataset_name}: Assignment 1 single-neuron perceptron baseline "
          f"(trained on the same 60% training split, evaluated on the same test split)")
    pairs = one_against_one_pairs(classes)
    perc_classifiers = {}
    for (a, b) in pairs:
        Xp, yp = subset_for_pair(X_train_n, y_train, a, b)
        target = np.where(yp == a, 0, 1)
        p = Perceptron(n_inputs=X.shape[1], activation='logistic',
                        learning_rate=0.2, n_epochs=2000, random_state=random_state)
        p.fit(Xp, target)
        perc_classifiers[(a, b)] = p

    def perceptron_predict_pair(clf, X, class_a, class_b):
        raw = clf.predict(X)
        return np.where(raw == 0, class_a, class_b)

    vote_matrix = np.empty((X_test_n.shape[0], len(pairs)), dtype=int)
    for k, (a, b) in enumerate(pairs):
        vote_matrix[:, k] = perceptron_predict_pair(perc_classifiers[(a, b)], X_test_n, a, b)
    y_pred_perceptron = aggregate_votes(vote_matrix, classes)

    perc_acc = accuracy(y_test, y_pred_perceptron)
    fcnn_acc = accuracy(y_test, y_pred_test)
    print(f"Assignment 1 perceptron test accuracy: {perc_acc:.4f}")
    print(f"Assignment 2 FCNN (best) test accuracy: {fcnn_acc:.4f}")

    return {
        'dataset': dataset_name,
        'best_hidden_dims': best['hidden_dims'],
        'best_activation': activation,
        'fcnn_test_accuracy': fcnn_acc,
        'perceptron_test_accuracy': perc_acc,
    }


def main():
    # Dataset 1: 1 hidden layer, try a few hidden-node counts
    summary1 = run_for_dataset(
        'dataset1',
        hidden_options=[(4,), (8,), (16,)],
        lr=0.1, n_epochs=200,
        out_subdir=f"{OUT_DIR}/dataset1")

    # Dataset 2: 2 hidden layers, try a few hidden-node combinations
    summary2 = run_for_dataset(
        'dataset2',
        hidden_options=[(8, 4), (16, 8), (16, 16)],
        lr=0.05, n_epochs=200,
        out_subdir=f"{OUT_DIR}/dataset2")

    print(f"\n{'=' * 70}")
    print("SUMMARY: FCNN (Assignment 2) vs single-neuron perceptron (Assignment 1)")
    print(f"{'=' * 70}")
    for s in [summary1, summary2]:
        print(f"{s['dataset']}: best hidden_dims={s['best_hidden_dims']}, "
              f"activation={s['best_activation']} | "
              f"FCNN test acc={s['fcnn_test_accuracy']:.4f} | "
              f"Perceptron test acc={s['perceptron_test_accuracy']:.4f}")

    print("\nDone. Plots saved under:", OUT_DIR)


if __name__ == '__main__':
    main()
