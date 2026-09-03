import os
import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork
from data_utils import train_test_split_per_class
from metrics import print_all_metrics
from plotting import plot_error_curve, plot_decision_region

OUT_DIR = "results/classification"
DATA_DIR = "data/Group06"


def load_dataset(name="dataset1"):
    if name == "dataset1":
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
        path = os.path.join(DATA_DIR, "Classification", "NLS_Group06.txt")
        X = np.loadtxt(path, skiprows=1)
        y = np.concatenate([np.full(500, 0), np.full(500, 1), np.full(1000, 2)])

    else:
        raise ValueError(f"unknown dataset name {name}")

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def preprocess_features(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    X = (X - mean) / std

    return X, y


def one_hot_encode(y, classes):
    y = np.asarray(y, dtype=int)
    classes = np.asarray(classes)
    class_to_index = {cls: i for i, cls in enumerate(classes)}
    encoded = np.zeros((len(y), len(classes)), dtype=float)
    for i, label in enumerate(y):
        encoded[i, class_to_index[int(label)]] = 1.0
    return encoded


def train_classifier(X_train, y_train, activation, lr=0.05, n_epochs=500, random_state=0):
    classes = sorted(np.unique(y_train))
    y_encoded = one_hot_encode(y_train, classes)

    model = NeuralNetwork(
        input_size=X_train.shape[1],
        net_dims=[X_train.shape[1], 4, 8, 4 , len(classes)],
        activation=activation,
        type='classification',
        seed=random_state,
        learning_rate=lr,
        weight_decay = 0,
        batch_size= 1,
        epochs = n_epochs

    )
    model.fit(X_train, y_encoded)
    return model, classes


def predict_multiclass(model, X, classes):
    probs = model.predict(X)
    pred_idx = np.argmax(probs, axis=1)
    return np.asarray(classes)[pred_idx]


def run_for_dataset(dataset_name, learning_rate, n_epochs, out_subdir):
    os.makedirs(out_subdir, exist_ok=True)
    X, y = load_dataset(dataset_name)
    X, y = preprocess_features(X, y)
    X_train, X_test, y_train, y_test = train_test_split_per_class(X, y, train_frac=0.7, random_state=0)
    classes = sorted(np.unique(y))

    for activation in ['logistic', 'tanh']:
        print(f"\n=== {dataset_name} | activation = {activation} ===")

        model, classes = train_classifier(X_train, y_train, activation=activation,
                                        lr=learning_rate, n_epochs=n_epochs, random_state=0)

        error_fig = plot_error_curve(model.error_history_,
                                    title=f"{dataset_name} training loss ({activation})",
                                    save_path=f"{out_subdir}/{dataset_name}_{activation}_loss.png")
        plt.close(error_fig)

        def predict_fn(grid, model=model, classes=classes):
            return predict_multiclass(model, grid, classes)

        region_fig = plot_decision_region(X_train, y_train, predict_fn,
                                         title=f"{dataset_name} decision region ({activation})",
                                         save_path=f"{out_subdir}/{dataset_name}_{activation}_region.png")
        plt.close(region_fig)

        y_pred_test = predict_multiclass(model, X_test, classes)
        print("Test set metrics:")
        print_all_metrics(y_test, y_pred_test, labels=classes)


def main():
    run_for_dataset('dataset1', learning_rate=0.002, n_epochs=200, out_subdir=f"{OUT_DIR}/dataset1")
    run_for_dataset('dataset2', learning_rate=0.002, n_epochs=1000, out_subdir=f"{OUT_DIR}/dataset2")
    print("\nDone. Plots saved under:", OUT_DIR)


if __name__ == '__main__':
    main()
