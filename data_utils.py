"""
data_utils.py

Loads the Group 06 MNIST-subset (5 classes: 0, 3, 4, 8, 9) from the
train/val/test folders of JPEG images, flattens each 28x28 image to a
784-dim vector, and returns everything as PyTorch tensors.

Group 06
"""

import os
import numpy as np
import torch
from PIL import Image

DATA_DIR = "data"
CLASS_FOLDERS = ['0', '3', '4', '8', '9']   # original MNIST digit labels
# map original digit labels -> 0..4 class indices for cross-entropy
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CLASS_FOLDERS)}
INDEX_TO_LABEL = {idx: label for idx, label in enumerate(CLASS_FOLDERS)}


def load_split(split_name):
    """
    split_name: 'train', 'val', or 'test'
    Returns X (N, 784) float32 tensor in [0, 1], y (N,) long tensor with
    class indices 0..4.
    """
    X_list = []
    y_list = []
    split_dir = os.path.join(DATA_DIR, split_name)

    for cls_folder in CLASS_FOLDERS:
        cls_dir = os.path.join(split_dir, cls_folder)
        for fname in sorted(os.listdir(cls_dir)):
            img = Image.open(os.path.join(cls_dir, fname))
            arr = np.asarray(img, dtype=np.float32) / 255.0   # (28, 28) in [0,1]
            X_list.append(arr.reshape(-1))                     # flatten to 784
            y_list.append(LABEL_TO_INDEX[cls_folder])

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)
    return X, y


def load_all():
    X_train, y_train = load_split('train')
    X_val, y_val = load_split('val')
    X_test, y_test = load_split('test')
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_all()
    print('train:', X_train.shape, y_train.shape)
    print('val:  ', X_val.shape, y_val.shape)
    print('test: ', X_test.shape, y_test.shape)
    print('class distribution (train):', torch.bincount(y_train))
    print('pixel value range:', X_train.min().item(), X_train.max().item())
