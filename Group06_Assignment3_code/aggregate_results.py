"""
aggregate_results.py

Loads all 21 (architecture, optimizer) result files and produces:
  1. A table of epochs-to-convergence per architecture x optimizer.
  2. Superimposed error-vs-epoch plots, one figure per architecture
     (all 7 optimizers on the same axes).
  3. A table of training and validation accuracy per architecture x optimizer.
  4. Selects the best architecture+optimizer combo by validation
     accuracy, then reports its test confusion matrix, test accuracy,
     training accuracy, and training confusion matrix.

Group 06
"""

import json
import os
import torch
import matplotlib.pyplot as plt

from model import FCNN
from data_utils import load_all, CLASS_FOLDERS
from optimizer_configs import ARCHITECTURES, OPTIMIZER_DISPLAY_NAMES

RESULTS_DIR = "results"
OUT_DIR = "results/summary"

OPT_ORDER = ['sgd', 'batch_gd', 'sgd_momentum', 'sgd_nag', 'adagrad', 'rmsprop', 'adam']


def load_all_results():
    results = {}
    for arch in ARCHITECTURES:
        for opt in OPT_ORDER:
            path = f"{RESULTS_DIR}/{arch}__{opt}.json"
            with open(path) as f:
                results[(arch, opt)] = json.load(f)
    return results


def print_convergence_table(results):
    print("\n" + "=" * 90)
    print("TABLE 1: Epochs to convergence (architecture x optimizer)")
    print("=" * 90)
    header = f"{'Optimizer':32s}" + "".join(f"{a:>12s}" for a in ARCHITECTURES)
    print(header)
    for opt in OPT_ORDER:
        row = f"{OPTIMIZER_DISPLAY_NAMES[opt]:32s}"
        for arch in ARCHITECTURES:
            r = results[(arch, opt)]
            tag = "*" if r.get('hit_max_epochs') else ""
            row += f"{str(r['epochs_trained']) + tag:>12s}"
        print(row)
    print("(* = did not meet the literal 1e-4 threshold within the practical "
          "epoch budget used; see notes in the report)")


def print_accuracy_table(results):
    print("\n" + "=" * 90)
    print("TABLE 2: Training / Validation accuracy (architecture x optimizer)")
    print("=" * 90)
    for arch in ARCHITECTURES:
        print(f"\n--- {arch} (hidden_dims={ARCHITECTURES[arch]}) ---")
        print(f"{'Optimizer':32s}{'Train Acc':>12s}{'Val Acc':>12s}")
        for opt in OPT_ORDER:
            r = results[(arch, opt)]
            print(f"{OPTIMIZER_DISPLAY_NAMES[opt]:32s}"
                  f"{r['train_accuracy']*100:>11.2f}%{r['val_accuracy']*100:>11.2f}%")


def plot_superimposed_error_curves(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    for arch in ARCHITECTURES:
        fig, ax = plt.subplots(figsize=(8, 6))
        for opt in OPT_ORDER:
            r = results[(arch, opt)]
            ax.plot(range(1, len(r['error_history']) + 1), r['error_history'],
                    label=OPTIMIZER_DISPLAY_NAMES[opt], linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average training error (cross-entropy)')
        ax.set_title(f'{arch} (hidden_dims={ARCHITECTURES[arch]}): all optimizers')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/{arch}_all_optimizers_error_curves.png", dpi=150)
        plt.close(fig)
        print(f"saved {OUT_DIR}/{arch}_all_optimizers_error_curves.png")


def confusion_matrix_5class(preds, y):
    cm = torch.zeros(5, 5, dtype=torch.int64)
    for t, p in zip(y, preds):
        cm[t, p] += 1
    return cm


def print_confusion_matrix(cm, labels):
    header = "        " + "  ".join(f"P{l}" for l in labels)
    print(header)
    for i, lab in enumerate(labels):
        row = "  ".join(f"{v:4d}" for v in cm[i].tolist())
        print(f"T{lab}     {row}")


def main():
    results = load_all_results()
    print_convergence_table(results)
    print_accuracy_table(results)
    plot_superimposed_error_curves(results)

    # ---- best architecture+optimizer by validation accuracy ----
    best_key = max(results, key=lambda k: results[k]['val_accuracy'])
    best_arch, best_opt = best_key
    best_result = results[best_key]
    print("\n" + "=" * 90)
    print(f"BEST COMBO: architecture={best_arch} (hidden_dims={ARCHITECTURES[best_arch]}), "
          f"optimizer={OPTIMIZER_DISPLAY_NAMES[best_opt]}")
    print(f"Validation accuracy: {best_result['val_accuracy']*100:.2f}%")
    print("=" * 90)

    # load the best model's weights and evaluate on train + test
    X_train, y_train, X_val, y_val, X_test, y_test = load_all()
    model = FCNN(784, ARCHITECTURES[best_arch], 5, activation='relu')
    model.load_state_dict(torch.load(f"{RESULTS_DIR}/{best_arch}__{best_opt}_weights.pt",
                                      weights_only=True))
    model.eval()

    with torch.no_grad():
        train_preds = model(X_train).argmax(dim=1)
        test_preds = model(X_test).argmax(dim=1)

    train_acc = (train_preds == y_train).float().mean().item()
    test_acc = (test_preds == y_test).float().mean().item()

    train_cm = confusion_matrix_5class(train_preds, y_train)
    test_cm = confusion_matrix_5class(test_preds, y_test)

    labels = CLASS_FOLDERS  # original digit labels: ['0','3','4','8','9']

    print(f"\nTraining accuracy: {train_acc*100:.2f}%")
    print("Training confusion matrix:")
    print_confusion_matrix(train_cm, labels)

    print(f"\nTest accuracy: {test_acc*100:.2f}%")
    print("Test confusion matrix:")
    print_confusion_matrix(test_cm, labels)

    # save a summary json for the report
    summary = {
        'best_arch': best_arch,
        'best_hidden_dims': ARCHITECTURES[best_arch],
        'best_optimizer': best_opt,
        'best_optimizer_display': OPTIMIZER_DISPLAY_NAMES[best_opt],
        'val_accuracy': best_result['val_accuracy'],
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_confusion_matrix': train_cm.tolist(),
        'test_confusion_matrix': test_cm.tolist(),
        'class_labels': labels,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/best_combo_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to {OUT_DIR}/best_combo_summary.json")
    print(f"Saved error-curve plots to {OUT_DIR}/")


if __name__ == '__main__':
    main()
