import copy

lookup = {}

def register(name):
    def decorator(cls):
        lookup[name] = cls
        return cls
    return decorator

def make(spec, args=None):
    if args is not None:
        _args = copy.deepcopy(spec['args'])
        _args.update(args)
    else:
        _args = copy.deepcopy(spec['args'])

    return lookup[spec['name']](**_args)

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
