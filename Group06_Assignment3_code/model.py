import torch
import torch.nn as nn


class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes):
        super().__init__()
        assert 3 <= len(hidden_dims) <= 5, "assignment requires 3 to 5 hidden layers"

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh()) 
            prev_dim = h

        layers.append(nn.Linear(prev_dim, n_classes)) # raw logits... nn.CrossEntropyLoss expects this supposedly

        self.net = nn.Sequential(*layers) #connects layers for forward pass

    def forward(self, x):
        return self.net(x)
