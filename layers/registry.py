import copy

lookup = {}

def register(name):
    def decorator(cls):
        lookup[name] = cls
        return cls
    return decorator

def make(layer_spec, args=None):
    if args is not None:
        layer_args = copy.deepcopy(layer_spec['args'])
        layer_args.update(args)
    else:
        layer_args = layer_spec['args']
    layer = lookup[layer_spec['name']](**layer_args)
    return layer

def create(name, **kwargs):
    if isinstance(name, str):
        if name in lookup:
            return make({"name": name, "args": kwargs})
        else:
            if len(kwargs):
                return eval(name)(**kwargs)
            else:
                return eval(name)
    raise ValueError(f"Must pass name of registered component or valid python code to `create` methods:\n'{name}'\nis invalid")