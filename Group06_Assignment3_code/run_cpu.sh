#!/bin/bash
arch="arch4_gradual"
for opt in batch_gd adagrad rmsprop; do
  python3 train_one.py "$arch" "$opt" --max_epochs 1500
done