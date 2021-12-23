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
            act_args = copy.deepcopy(act_spec['args'])
            act_args.update(args)
        else:
            act_args = act_spec['args']
        act = lookup[act_spec['name']](**act_args)
    else:
        act = lookup[act_spec['name']]()
    return act