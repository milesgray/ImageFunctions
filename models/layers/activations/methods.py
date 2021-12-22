import torch

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