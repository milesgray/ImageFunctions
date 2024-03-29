import copy

from torch.optim import SGD, Adam, AdamW, NAdam, RAdam

lookup = {
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW,
    'nadam': NAdam,
    'radam': RAdam
}

def register(name):
    def decorator(cls):
        lookup[name] = cls
        return cls
    return decorator

def make(param_list, spec, args=None):
    if args is not None:
        target_args = copy.deepcopy(spec['args'])
        target_args.update(args)
    else:
        target_args = spec['args']
    target = lookup[spec['name']](param_list, **target_args)
    return target
