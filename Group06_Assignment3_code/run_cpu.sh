#!/bin/bash
arch="arch4_gradual"
for opt in sgd sgd_momentum sgd_nag adam; do
  python3 train_one.py "$arch" "$opt" --max_epochs 1500
done