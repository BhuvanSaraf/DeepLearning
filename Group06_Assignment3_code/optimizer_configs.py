"""
optimizer_configs.py

The 7 optimizer configurations required by the assignment, with the
exact hyperparameters specified in the assignment PDF:
  - learning rate = 0.001 for all optimizers
  - momentum = 0.9 for generalized delta rule and NAG
  - RMSProp: beta=0.99, eps=1e-8
  - Adam: beta1=0.9, beta2=0.999, eps=1e-8

Each entry also specifies the batch size mode:
  - 'sgd'  -> batch_size = 1  (true stochastic gradient descent)
  - 'full' -> batch_size = N  (the full training set, i.e. batch/vanilla GD)

Group 06
"""

import torch


def get_optimizer_configs():
    return {
        'sgd':               {'batch_mode': 'sgd',  'make': lambda params: torch.optim.SGD(params, lr=0.001, momentum=0.0)},
        'batch_gd':           {'batch_mode': 'full', 'make': lambda params: torch.optim.SGD(params, lr=0.001, momentum=0.0)},
        'sgd_momentum':       {'batch_mode': 'sgd',  'make': lambda params: torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=False)},
        'sgd_nag':            {'batch_mode': 'sgd',  'make': lambda params: torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)},
        'adagrad':            {'batch_mode': 'full', 'make': lambda params: torch.optim.Adagrad(params, lr=0.001)},
        'rmsprop':            {'batch_mode': 'full', 'make': lambda params: torch.optim.RMSprop(params, lr=0.001, alpha=0.99, eps=1e-8)},
        'adam':               {'batch_mode': 'sgd',  'make': lambda params: torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8)},
    }


# human-readable names for plots/tables/report
OPTIMIZER_DISPLAY_NAMES = {
    'sgd': 'SGD (batch_size=1)',
    'batch_gd': 'Batch GD (batch_size=N)',
    'sgd_momentum': 'SGD + Momentum (Generalized Delta)',
    'sgd_nag': 'SGD + Nesterov (NAG)',
    'adagrad': 'AdaGrad (batch_size=N)',
    'rmsprop': 'RMSProp (batch_size=N)',
    'adam': 'Adam (batch_size=1)',
}


ARCHITECTURES = {
    'arch3': [256, 128, 64],           # 3 hidden layers
    'arch4': [256, 128, 64, 32],       # 4 hidden layers
    'arch5': [256, 128, 64, 32, 16],   # 5 hidden layers
}
