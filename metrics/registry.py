import copy


lookup ={
}

def register(name):
    def decorator(cls):
        lookup[name] = cls
        return cls
    return decorator

def make(spec, args=None):
    if args is not None:
        target_args = copy.deepcopy(spec['args'])
        target_args.update(args)
    else:
        target_args = spec['args']
    target = lookup[spec['name']](**target_args)
    return target
