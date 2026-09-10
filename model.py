"""
model.py

A configurable fully-connected neural network (3-5 hidden layers,
per the assignment). Cross-entropy loss is applied outside the model
(via nn.CrossEntropyLoss, which expects raw logits), so the final
layer here has no activation.

Group 06
"""

import torch
import torch.nn as nn


class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes, activation='relu'):
        super().__init__()
        assert 3 <= len(hidden_dims) <= 5, "assignment requires 3 to 5 hidden layers"

        act_layer = {'relu': nn.ReLU, 'tanh': nn.Tanh}[activation]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_layer())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_classes))  # raw logits, no softmax

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
