import copy

lookup = {}

def register(name):
    def decorator(cls):
        lookup[name] = cls
        return cls
    return decorator

def make(act_spec, args=None):
    if 'args' in act_spec:
        if args is not None:
            layer_args = copy.deepcopy(layer_spec['args'])
            layer_args.update(args)
        else:
            layer_args = layer_spec['args']
        layer = lookup[layer_spec['name']](**layer_args)
    else:
        layer = lookup[layer_spec['name']]()
    return layer