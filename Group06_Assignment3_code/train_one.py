"""
train_one.py

Trains ONE (architecture, optimizer) combination and saves the
results to disk. Designed to be called repeatedly/resumably: if the
result file for a combo already exists, it is skipped.

Usage:
    python3 train_one.py <arch_name> <optimizer_name> [--max_epochs N]

Every architecture uses the SAME initial weights across all 7
optimizers (per the assignment's requirement (d)). This is done by
saving each architecture's initial state_dict the first time it's
needed, and reusing it for every optimizer trained on that
architecture.

Group 06
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn

from data_utils import load_all
from model import FCNN
from optimizer_configs import get_optimizer_configs, ARCHITECTURES

RESULTS_DIR = "results"
INIT_DIR = "results/init_weights"
TOL = 1e-4  # assignment's stopping criterion: |avg_error(t) - avg_error(t-1)| < 1e-4


def get_or_create_init_weights(arch_name, hidden_dims, seed=0):
    os.makedirs(INIT_DIR, exist_ok=True)
    path = f"{INIT_DIR}/{arch_name}_init.pt"
    if os.path.exists(path):
        return torch.load(path, weights_only=True)
    torch.manual_seed(seed)
    model = FCNN(784, hidden_dims, 5, activation='relu')
    state = model.state_dict()
    torch.save(state, path)
    return state


def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item()


def confusion_matrix_5class(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
    cm = torch.zeros(5, 5, dtype=torch.int64)
    for t, p in zip(y, preds):
        cm[t, p] += 1
    return cm.tolist()


def train_one(arch_name, optimizer_name, max_epochs):
    hidden_dims = ARCHITECTURES[arch_name]
    opt_cfg = get_optimizer_configs()[optimizer_name]

    out_path = f"{RESULTS_DIR}/{arch_name}__{optimizer_name}.json"
    ckpt_path = f"{RESULTS_DIR}/{arch_name}__{optimizer_name}_ckpt.pt"

    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists")
        return

    X_train, y_train, X_val, y_val, X_test, y_test = load_all()
    n = X_train.shape[0]

    model = FCNN(784, hidden_dims, 5, activation='relu')
    optimizer = opt_cfg['make'](model.parameters())
    batch_size = 1 if opt_cfg['batch_mode'] == 'sgd' else n
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    error_history = []
    prev_avg_loss = None
    elapsed_so_far = 0.0

    if os.path.exists(ckpt_path):
        # resume from a checkpoint left by a previous (timed-out) run
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        error_history = ckpt['error_history']
        prev_avg_loss = ckpt['prev_avg_loss']
        start_epoch = ckpt['epoch']
        elapsed_so_far = ckpt['elapsed_so_far']
        print(f"[resume] {arch_name}/{optimizer_name} from epoch {start_epoch}")
    else:
        init_state = get_or_create_init_weights(arch_name, hidden_dims)
        model.load_state_dict(init_state)   # SAME initial weights per architecture

    t0 = time.time()
    converged = False

    for epoch in range(start_epoch, max_epochs):
        model.train()
        perm = torch.randperm(n)
        Xs, ys = X_train[perm], y_train[perm]

        total_loss = 0.0
        for start in range(0, n, batch_size):
            end = start + batch_size
            xb, yb = Xs[start:end], ys[start:end]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / n
        error_history.append(avg_loss)

        # assignment's literal stopping criterion (e): absolute
        # difference between average error of successive epochs < 1e-4.
        # No patience/smoothing -- deliberately literal, per the spec.
        if prev_avg_loss is not None and abs(prev_avg_loss - avg_loss) < TOL:
            converged = True
            epoch_reached = epoch + 1
            break
        prev_avg_loss = avg_loss

        # checkpoint after every epoch so a timeout doesn't lose progress
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'error_history': error_history,
            'prev_avg_loss': prev_avg_loss,
            'epoch': epoch + 1,
            'elapsed_so_far': elapsed_so_far + (time.time() - t0),
        }, ckpt_path)
    else:
        epoch_reached = max_epochs

    elapsed = elapsed_so_far + (time.time() - t0)
    epochs_trained = len(error_history)

    train_acc = accuracy(model, X_train, y_train)
    val_acc = accuracy(model, X_val, y_val)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        'arch_name': arch_name,
        'hidden_dims': hidden_dims,
        'optimizer_name': optimizer_name,
        'batch_size': batch_size,
        'epochs_trained': epochs_trained,
        'max_epochs': max_epochs,
        'hit_max_epochs': not converged,
        'error_history': error_history,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'elapsed_seconds': elapsed,
    }
    with open(out_path, 'w') as f:
        json.dump(result, f)

    # also save the trained model weights, in case this turns out to be
    # the best (architecture, optimizer) combo and we need test-set
    # metrics for it later
    torch.save(model.state_dict(), f"{RESULTS_DIR}/{arch_name}__{optimizer_name}_weights.pt")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print(f"[done] {arch_name} / {optimizer_name}: "
          f"epochs={epochs_trained}/{max_epochs}, "
          f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
          f"time={elapsed:.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arch_name', choices=list(ARCHITECTURES.keys()))
    parser.add_argument('optimizer_name', choices=list(get_optimizer_configs().keys()))
    parser.add_argument('--max_epochs', type=int, default=1500)
    args = parser.parse_args()

    train_one(args.arch_name, args.optimizer_name, args.max_epochs)
