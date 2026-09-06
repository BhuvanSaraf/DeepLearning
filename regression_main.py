"""
regression_main.py

Assignment 2 regression pipeline:
  - loads Dataset 1 (univariate) and Dataset 2 (bivariate)
  - splits into train/val/test (60/20/20)
  - trains an FCNN (from scratch, squared error loss, SGD) --
    1 hidden layer for Dataset 1; BOTH 1 and 2 hidden layers tried
    for Dataset 2, with several hidden-node counts
  - reports RMSE/%RMSE on train AND validation for every architecture
    tried (a short/cheap search pass), picks the best by validation
    RMSE, retrains it for longer, then reports error-vs-epoch,
    model-vs-target plots, scatter plots, and node-output plots for
    train/val/test, plus test RMSE/%RMSE, for that architecture only
  - compares the best FCNN against the Assignment 1 linear perceptron
    baseline on the same train/test split

Group 06
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork
from metrics import rmse, percent_rmse
from plotting import (plot_error_curve, plot_model_vs_target_1d,
                       plot_model_vs_target_2d, plot_scatter_target_vs_pred,
                       plot_node_outputs)
from perceptron_baseline import Perceptron

OUT_DIR = "results/regression"
DATA_DIR = "data/Group06"


def load_dataset(name="dataset1"):
    if name == "dataset1":
        path = os.path.join(DATA_DIR, "Regression", "UnivariateData", "6.csv")
        data = np.loadtxt(path, delimiter=",")
        x = data[:, 0].reshape(-1, 1)
        y = data[:, 1]
        return x, y

    elif name == "dataset2":
        path = os.path.join(DATA_DIR, "Regression", "BivariateData", "6.csv")
        data = np.loadtxt(path, delimiter=",")
        X = data[:, :2]
        y = data[:, 2]
        return X, y

    else:
        raise ValueError(f"unknown dataset name {name}")


def train_val_test_split_regression(X, y, train_frac=0.6, val_frac=0.2, random_state=0):
    np.random.seed(random_state)
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return (X[train_idx], X[val_idx], X[test_idx],
            y[train_idx], y[val_idx], y[test_idx])


def train_fcnn(X_train, y_train, input_size, hidden_dims, lr, n_epochs, random_state=0):
    net_dims = [input_size] + list(hidden_dims) + [1]
    model = NeuralNetwork(
        input_size=input_size,
        net_dims=net_dims,
        activation='tanh',
        type='regression',
        seed=random_state,
        learning_rate=lr,
        weight_decay=0,
        batch_size=1,     # true SGD, one sample at a time
        epochs=n_epochs,
    )
    model.fit(X_train, y_train)
    return model


def run_for_dataset(dataset_name, n_inputs, hidden_options, lr,
                     search_epochs, final_epochs, out_subdir, random_state=0):
    os.makedirs(out_subdir, exist_ok=True)

    X, y = load_dataset(dataset_name)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_regression(
        X, y, train_frac=0.6, val_frac=0.2, random_state=random_state)

    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    x_std[x_std == 0] = 1.0
    X_train_n = (X_train - x_mean) / x_std
    X_val_n = (X_val - x_mean) / x_std
    X_test_n = (X_test - x_mean) / x_std

    print(f"\n{'=' * 70}")
    print(f"{dataset_name}: architecture search")
    print(f"{'=' * 70}")

    search_results = []
    for hidden_dims in hidden_options:
        model = train_fcnn(X_train_n, y_train, n_inputs, hidden_dims, lr,
                            search_epochs, random_state)
        y_pred_train = model.predict(X_train_n)
        y_pred_val = model.predict(X_val_n)

        rmse_train = rmse(y_train, y_pred_train)
        pct_train = percent_rmse(y_train, y_pred_train)
        rmse_val = rmse(y_val, y_pred_val)
        pct_val = percent_rmse(y_val, y_pred_val)

        print(f"hidden_dims={hidden_dims}: "
              f"Train RMSE={rmse_train:.4f} (%={pct_train:.2f}%) | "
              f"Val RMSE={rmse_val:.4f} (%={pct_val:.2f}%)")

        search_results.append({
            'hidden_dims': hidden_dims,
            'val_rmse': rmse_val,
            'val_pct_rmse': pct_val,
            'train_rmse': rmse_train,
        })

    best = min(search_results, key=lambda r: r['val_rmse'])
    print(f"\n>>> Best architecture for {dataset_name}: "
          f"hidden_dims={best['hidden_dims']} (val RMSE={best['val_rmse']:.4f})")

    # retrain the best architecture for longer for the final, reported model
    best_model = train_fcnn(X_train_n, y_train, n_inputs, best['hidden_dims'],
                             lr, final_epochs, random_state)

    y_pred_train = best_model.predict(X_train_n)
    y_pred_val = best_model.predict(X_val_n)
    y_pred_test = best_model.predict(X_test_n)

    rmse_test = rmse(y_test, y_pred_test)
    pct_test = percent_rmse(y_test, y_pred_test)
    print(f"\n{dataset_name}: TEST SET performance for best architecture "
          f"(hidden_dims={best['hidden_dims']})")
    print(f"Test RMSE = {rmse_test:.4f}  (%RMSE = {pct_test:.2f}%)")

    # error vs epoch (best architecture only)
    fig = plot_error_curve(best_model.error_history_,
                            title=f"{dataset_name}: training error (hidden_dims={best['hidden_dims']})",
                            save_path=f"{out_subdir}/{dataset_name}_best_error_curve.png")
    plt.close(fig)

    # model output vs target, and scatter, for train/val/test
    splits = [('train', X_train, X_train_n, y_train, y_pred_train),
              ('val', X_val, X_val_n, y_val, y_pred_val),
              ('test', X_test, X_test_n, y_test, y_pred_test)]

    for split_name, X_raw, X_n, y_true, y_pred in splits:
        if n_inputs == 1:
            fig = plot_model_vs_target_1d(X_raw[:, 0], y_true, y_pred,
                                           title=f"{dataset_name}: {split_name} (model vs target)",
                                           save_path=f"{out_subdir}/{dataset_name}_{split_name}_fit.png")
        else:
            fig = plot_model_vs_target_2d(X_raw[:, 0], X_raw[:, 1], y_true, y_pred,
                                           title=f"{dataset_name}: {split_name} (model vs target)",
                                           save_path=f"{out_subdir}/{dataset_name}_{split_name}_fit.png")
        plt.close(fig)

        fig = plot_scatter_target_vs_pred(y_true, y_pred,
                                           title=f"{dataset_name}: {split_name} target vs model output",
                                           save_path=f"{out_subdir}/{dataset_name}_{split_name}_scatter.png")
        plt.close(fig)

    # hidden / output node plots for train, val, test
    node_dir = f"{out_subdir}/{dataset_name}_node_outputs"
    for split_name, _, X_n, _, _ in splits:
        plot_node_outputs(best_model, X_n, input_dim=n_inputs,
                           title_prefix=dataset_name, out_dir=node_dir, split_name=split_name)

    # ---- comparison against the Assignment 1 linear perceptron baseline ----
    print(f"\n{dataset_name}: Assignment 1 linear perceptron baseline "
          f"(trained on the same 60% training split, evaluated on the same test split)")
    perc = Perceptron(n_inputs=n_inputs, activation='linear', learning_rate=0.05,
                       n_epochs=500, random_state=random_state)
    perc.fit(X_train_n, y_train)
    y_pred_perc_test = perc.predict(X_test_n)
    perc_rmse = rmse(y_test, y_pred_perc_test)
    perc_pct = percent_rmse(y_test, y_pred_perc_test)
    print(f"Assignment 1 perceptron test RMSE = {perc_rmse:.4f}  (%RMSE = {perc_pct:.2f}%)")
    print(f"Assignment 2 FCNN (best)  test RMSE = {rmse_test:.4f}  (%RMSE = {pct_test:.2f}%)")

    return {
        'dataset': dataset_name,
        'best_hidden_dims': best['hidden_dims'],
        'fcnn_test_rmse': rmse_test,
        'fcnn_test_pct_rmse': pct_test,
        'perceptron_test_rmse': perc_rmse,
        'perceptron_test_pct_rmse': perc_pct,
    }


def main():
    # Dataset 1: 1 hidden layer, try a few hidden-node counts
    summary1 = run_for_dataset(
        'dataset1', n_inputs=1,
        hidden_options=[(4,), (8,), (16,)],
        lr=0.02, search_epochs=150, final_epochs=500,
        out_subdir=f"{OUT_DIR}/dataset1")

    # Dataset 2: try BOTH 1 hidden layer and 2 hidden layers
    summary2 = run_for_dataset(
        'dataset2', n_inputs=2,
        hidden_options=[(8,), (16,), (8, 8), (16, 8)],
        lr=0.02, search_epochs=60, final_epochs=300,
        out_subdir=f"{OUT_DIR}/dataset2")

    print(f"\n{'=' * 70}")
    print("SUMMARY: FCNN (Assignment 2) vs linear perceptron (Assignment 1)")
    print(f"{'=' * 70}")
    for s in [summary1, summary2]:
        print(f"{s['dataset']}: best hidden_dims={s['best_hidden_dims']} | "
              f"FCNN test RMSE={s['fcnn_test_rmse']:.4f} (%={s['fcnn_test_pct_rmse']:.2f}%) | "
              f"Perceptron test RMSE={s['perceptron_test_rmse']:.4f} (%={s['perceptron_test_pct_rmse']:.2f}%)")

    print("\nDone. Plots saved under:", OUT_DIR)


if __name__ == '__main__':
    main()
