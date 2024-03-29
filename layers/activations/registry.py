import copy
import torch.nn as nn

lookup = {
    "elu": nn.ELU,
    "hardshrink": nn.Hardshrink,
    "hard_shrink": nn.Hardshrink,
    "hardsigmoid": nn.Hardsigmoid,
    "hard_sigmoid": nn.Hardsigmoid,
    "hardtanh": nn.Hardtanh,
    "hard_tanh": nn.Hardtanh,
    "hardswish": nn.Hardswish,
    "hard_swish": nn.Hardswish,
    "leakyrelu": nn.LeakyReLU,
    "leaky_relu": nn.LeakyReLU,
    "logsigmoid": nn.LogSigmoid,
    "prelu": nn.PReLU,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "rrelu": nn.RReLU,
    "selu": nn.SELU,
    "celu": nn.CELU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "softplus": nn.Softplus,
    "softshrink": nn.Softshrink,
    "softsign": nn.Softsign,
    "tanh": nn.Tanh,
    "tanhshrink": nn.Tanhshrink,
    "threshold": nn.Threshold,
    "glu": nn.GLU
}

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

def create(name, **kwargs):
    return make({"name": name, "args": kwargs})