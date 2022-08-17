import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

@torch.jit.script
def mish(x):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    """
    return x * torch.tanh(F.softplus(x))

@torch.jit.script
def logcosh(x):
    return torch.cosh(x + 1e-12).log()

@torch.jit.script
def xtanh(x):
    return torch.tanh(x) * x

@torch.jit.script
def xsigmoid(x):
    y = 1 + torch.exp(-x)
    y = torch.abs(y - x)
    z = 2 * y / x
    return z

@torch.jit.script
def centeredsigmoid(x):
    y = torch.sigmoid(x)
    y = y * 2
    y = y - 1
    return y

@torch.jit.script
def unitcenteredsigmoid(x):
    y = torch.sigmoid(x)
    y = y * 2
    return y

@torch.jit.script
def unitcenteredtanh(x):
    return torch.tanh(x) + 1

@torch.jit.script
def normalizer(x):
    mean = (x - x.mean(dim=1, keepdim=True)) 
    std = x.std(dim=1, keepdim=True)
    return mean / std

@torch.jit.script
def gaussian(x, a):
    return torch.exp(-x**2/(2*a**2))

@torch.jit.script
def hat(x):
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    ge_two = x.ge(ones * 2)
    lt_zero = x.lt(zeros)
    ge_zero = x.ge(torch.zeros_like(x))
    lt_one = x.lt(torch.ones_like(x))
    to_zero = torch.logical_or(ge_two, lt_zero)
    to_x = torch.logical_and(ge_zero, lt_one)
    to_2_minus = torch.logical_not(torch.logical_and(to_zero, to_x))
    x[to_zero] = 0
    x[to_2_minus] = 2 - x[to_2_minus]
    return x