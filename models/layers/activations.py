import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .registry import register


@torch.jit.script
def mish(x):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return x * torch.tanh(F.softplus(x))

@register("act_mish")
class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return mish(input)

@torch.jit.script
def logcosh(x):
    return torch.cosh(x + 1e-12).log()
@register("act_logcosh")
class LogCosh(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return logcosh(x)

@torch.jit.script
def xtanh(x):
    return torch.tanh(x) * x
@register("act_xtanh")
class XTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return xtanh(x)

@torch.jit.script
def xsigmoid(x):
    y = 1 + torch.exp(-x)
    y = torch.abs(y - x)
    z = 2 * y / x
    return z
@register("act_xsigmoid")
class XSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return xsigmoid(x)

@register("act_centered_sigmoid")
class CenteredSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid() * 2 - 1

@register("act_unit_centered_sigmoid")
class UnitCenteredSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid() * 2

@register("act_centered_tanh")
class UnitCenteredTanh(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh() + 1

@register("act_sine")
class Sine(nn.Module):
    def __init__(self, scale: float=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        return x.sin()

@register("act_normalizer")
class Normalizer(nn.Module):
    """
    Just normalizes its input
    """
    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2, f"Wrong shape: {x.shape}"

        return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)

@register("act_normal_clip")
class NormalClip(nn.Module):
    """
    Clips input values into [-2, 2] region
    """
    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x, -2, 2)

@register("act_sines_cosines")
class SinesCosines(nn.Module):
    """
    Sines-cosines activation function
    It applies both sines and cosines and concatenates the results
    """
    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([x.sin(), x.cos()], dim=1)

@register("act_scaled_leaky_relu")
class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope: float=0.2, scale: float=None):
        super().__init__()

        self.negative_slope = negative_slope
        self.scale = math.sqrt(2 / (1 + negative_slope ** 2)) if scale is None else scale

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * self.scale
     