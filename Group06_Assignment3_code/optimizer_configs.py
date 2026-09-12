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
    'arch3_gradual': [512, 128, 32],                    #TODO
    'arch4_gradual': [512, 256, 128, 32],                #....These are questionable...     
    'arch5_gradual': [512, 256, 128, 64, 32],            #we will change them shortly...
    'arch3_const':   [256, 256, 256],

}
input = 784