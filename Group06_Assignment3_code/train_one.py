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
TOLARENCE = 1e-4  #stopping criterion: |avg_error(t) - avg_error(t-1)| < 1e-4
INPUT_D = 784
OUTPUT_D = 5


def get_or_create_init_weights(arch_name,  hidden_dims, seed=0):
    os.makedirs(INIT_DIR, exist_ok=True)
    path = f"{INIT_DIR}/{arch_name}_init.pt"
    if os.path.exists(path):
        return torch.load(path, weights_only=True)
    torch.manual_seed(seed)
    model = FCNN(INPUT_D, hidden_dims, OUTPUT_D)
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
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    hidden_dims = ARCHITECTURES[arch_name]
    opt_cfg = get_optimizer_configs()[optimizer_name]

    out_path = f"{RESULTS_DIR}/{arch_name}__{optimizer_name}.json"

    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists")
        return

    X_train, y_train, X_val, y_val, X_test, y_test = load_all()
    N = X_train.shape[0]
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    X_val = X_val.to(DEVICE)
    y_val = y_val.to(DEVICE)
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)


    model = FCNN(INPUT_D, hidden_dims, OUTPUT_D)
    optimizer = opt_cfg['make'](model.parameters())
    batch_size = 1 if opt_cfg['batch_mode'] == 'sgd' else N
    criterion = nn.CrossEntropyLoss()

    error_history_t = []
    error_history_v = []
    prev_avg_loss = None

    init_state = get_or_create_init_weights(arch_name, hidden_dims)
    model.load_state_dict(init_state)   # SAME initial weights
    model = FCNN(INPUT_D, hidden_dims, OUTPUT_D).to(DEVICE)
    
    #------------------------------------------------------------------
    # To Evaluate 0th epoch errors, to ensure fairness
    model.eval()
    with torch.no_grad():
        train_out = model(X_train)
        train_loss = criterion(train_out, y_train)
    error_history_t.append(train_loss.item()) 
    with torch.no_grad():
        val_out = model(X_val)
        val_loss = criterion(val_out, y_val)
    error_history_v.append(val_loss.item()) 
    #------------------------------------------------------------------

    t0 = time.time()
    converged = False

    torch.manual_seed(0)  # fixed seed for reproducible shuffling

    for epoch in range(0 , max_epochs):
        model.train()
        perm = torch.randperm(N)
        Xs, ys = X_train[perm], y_train[perm]

        total_loss_t = 0.0
        for start in range(0, N, batch_size):
            end = start + batch_size
            xb, yb = Xs[start:end], ys[start:end]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss_t += loss.item() * xb.size(0)
        avg_loss_t = total_loss_t / N
        error_history_t.append(avg_loss_t)

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val)
        error_history_v.append(val_loss.item()) 

        # assignment's literal stopping criterion 
        if prev_avg_loss is not None and abs(prev_avg_loss - avg_loss_t) < TOLARENCE:
            converged = True
            epoch_reached = epoch + 1
            break
        prev_avg_loss = avg_loss_t

    else:
        epoch_reached = max_epochs

    elapsed =  (time.time() - t0)
    epochs_trained = len(error_history_t)

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
        'error_history_t': error_history_t,
        'error_history_v': error_history_v,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'elapsed_seconds': elapsed,     #json stuff
    }
    with open(out_path, 'w') as f:
        json.dump(result, f)

    #save the trained model weights, in case this turns out to be the best
    torch.save(model.state_dict(), f"{RESULTS_DIR}/{arch_name}__{optimizer_name}_weights.pt")

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
