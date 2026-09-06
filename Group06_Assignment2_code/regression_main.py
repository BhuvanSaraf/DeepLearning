import os
import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork
from metrics import rmse, percent_rmse
from plotting import (plot_error_curve, plot_model_vs_target_1d,
                       plot_model_vs_target_2d, plot_scatter_target_vs_pred)

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


def train_test_split_regression(X, y, train_frac=0.7, random_state=0):
    np.random.seed(random_state)
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(train_frac * n)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def run_for_dataset(dataset_name, n_inputs, net_dims, learning_rate, n_epochs, out_subdir):
    os.makedirs(out_subdir, exist_ok=True)
    X, y = load_dataset(dataset_name)
    X_train, X_t, y_train, y_t = train_test_split_regression(X, y, train_frac=0.6, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split_regression(X_t, y_t, train_frac=0.5, random_state=0)
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std == 0] = 1.0
    X_train_n = (X_train - x_mean) / x_std
    X_test_n = (X_test - x_mean) / x_std
    X_val_n = (X_val - x_mean) / x_std

    print(f"\n=== {dataset_name} (linear activation) ===")
    #clf = Perceptron(n_inputs=n_inputs, activation='linear', learning_rate=learning_rate,
    #                 n_epochs=n_epochs, random_state=0)
    
    clf = NeuralNetwork(
            input_size = n_inputs,
            net_dims = net_dims,
            activation = "tanh",
            type = 'regression', 
            seed = 0,
            learning_rate=learning_rate,
            weight_decay = 0,
            epochs = n_epochs
        )
    clf.fit(X_train_n, y_train)

    y_pred_train = clf.predict(X_train_n)
    y_pred_test = clf.predict(X_test_n)

    rmse_train = rmse(y_train, y_pred_train)
    pct_rmse_train = percent_rmse(y_train, y_pred_train)
    rmse_test = rmse(y_test, y_pred_test)
    pct_rmse_test = percent_rmse(y_test, y_pred_test)

    print(f"Train RMSE = {rmse_train:.4f}  (%RMSE = {pct_rmse_train:.2f}%)")
    print(f"Test  RMSE = {rmse_test:.4f}  (%RMSE = {pct_rmse_test:.2f}%)")

    # error vs epoch
    fig = plot_error_curve(clf.error_history_, title=f"{dataset_name}: Error vs Epoch",
                            save_path=f"{out_subdir}/{dataset_name}_error_curve.png")
    plt.close(fig)

    if n_inputs == 1:
        fig = plot_model_vs_target_1d(X_train[:, 0], y_train, y_pred_train,
                                       title=f"{dataset_name}: Train (model vs target)",
                                       save_path=f"{out_subdir}/{dataset_name}_train_fit.png")
        plt.close(fig)
        fig = plot_model_vs_target_1d(X_test[:, 0], y_test, y_pred_test,
                                       title=f"{dataset_name}: Test (model vs target)",
                                       save_path=f"{out_subdir}/{dataset_name}_test_fit.png")
        plt.close(fig)
    else:
        fig = plot_model_vs_target_2d(X_train[:, 0], X_train[:, 1], y_train, y_pred_train,
                                       title=f"{dataset_name}: Train (model vs target)",
                                       save_path=f"{out_subdir}/{dataset_name}_train_fit.png")
        plt.close(fig)
        fig = plot_model_vs_target_2d(X_test[:, 0], X_test[:, 1], y_test, y_pred_test,
                                       title=f"{dataset_name}: Test (model vs target)",
                                       save_path=f"{out_subdir}/{dataset_name}_test_fit.png")
        plt.close(fig)

    # scatter: target vs model output
    fig = plot_scatter_target_vs_pred(y_train, y_pred_train,
                                       title=f"{dataset_name}: Train target vs model output",
                                       save_path=f"{out_subdir}/{dataset_name}_train_scatter.png")
    plt.close(fig)
    fig = plot_scatter_target_vs_pred(y_test, y_pred_test,
                                       title=f"{dataset_name}: Test target vs model output",
                                       save_path=f"{out_subdir}/{dataset_name}_test_scatter.png")
    plt.close(fig)


def main():
    run_for_dataset('dataset1', n_inputs=1, net_dims=[1, 5, 1], learning_rate=0.02, n_epochs=500, out_subdir=f"{OUT_DIR}/dataset1")
    run_for_dataset('dataset2', n_inputs=2, net_dims=[2, 10, 1], learning_rate=0.02, n_epochs=500, out_subdir=f"{OUT_DIR}/dataset2")
    print("\nDone. Plots saved under:", OUT_DIR)


if __name__ == '__main__':
    main()
