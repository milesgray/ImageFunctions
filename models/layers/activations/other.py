import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .learnable import Scale, Balance

from .registry import register

@torch.jit.script
def mish(x):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    """
    return x * torch.tanh(F.softplus(x))
@register("mish")
class Mish(nn.Module):
    """
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
    """
    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return mish(input)

@torch.jit.script
def logcosh(x):
    return torch.cosh(x + 1e-12).log()
@register("logcosh")
class LogCosh(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return logcosh(x)

@torch.jit.script
def xtanh(x):
    return torch.tanh(x) * x
@register("xtanh")
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
@register("xsigmoid")
class XSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return xsigmoid(x)

@torch.jit.script
def centeredsigmoid(x):
    y = torch.sigmoid(x)
    y = y * 2
    y = y - 1
    return y
@register("centered_sigmoid")
class CenteredSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return centeredsigmoid(x)

@torch.jit.script
def unitcenteredsigmoid(x):
    y = torch.sigmoid(x)
    y = y * 2
    return y
@register("unit_centered_sigmoid")
class UnitCenteredSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return unitcenteredsigmoid(x)

@torch.jit.script
def unitcenteredtanh(x):
    return torch.tanh(x) + 1
@register("centered_tanh")
class UnitCenteredTanh(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return unitcenteredtanh(x)

@register("sine")
class Sine(nn.Module):
    def __init__(self, scale: float=1.0, learnable: bool=False):
        super().__init__()
        self.scale = Scale(scale) if learnable else scale

    def forward(self, x: Tensor) -> Tensor:
        return x.sin()

@torch.jit.script
def normalizer(x):
    mean = (x - x.mean(dim=1, keepdim=True)) 
    std = x.std(dim=1, keepdim=True)
    return mean / std
@register("normalizer")
class Normalizer(nn.Module):
    """
    Just normalizes its input
    """
    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2, f"Wrong shape: {x.shape}"

        return normalizer(x)

@register("normal_clip")
class NormalClip(nn.Module):
    """
    Clips input values into [-2, 2] region
    """
    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x, -2, 2)

@register("sines_cosines")
class SinesCosines(nn.Module):
    """
    Sines-cosines activation function
    It applies both sines and cosines and concatenates the results
    """
    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([x.sin(), x.cos()], dim=1)

@register("scaled_leaky_relu")
class ScaledLeakyReLU(nn.Module):
    def __init__(self, 
                 negative_slope: float=0.2, 
                 scale: float=None,
                 learnable: bool=False):
        super().__init__()        
        self.learnable = learnable
        self.negative_slope = nn.Parameter(torch.FloatTensor([negative_slope]), 
                                           requires_grad=self.learnable)
        
        scale = math.sqrt(2 / (1 + negative_slope ** 2)) if scale is None else scale
        self.scale = nn.Parameter(torch.FloatTensor([scale]), requires_grad=True)

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * self.scale
    
@register("adaptive_leaky_relu")
class AdaptiveLeakyReLU(nn.Module):
    def __init__(self, negative_slope: float=0.2, scale: float=None):
        super().__init__()

        self.negative_slope = nn.Parameter(torch.FloatTensor([negative_slope]), requires_grad=True)
        scale = math.sqrt(2 / (1 + negative_slope ** 2)) if scale is None else scale
        self.scale = nn.Parameter(torch.FloatTensor([scale]), requires_grad=True)

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=self.negative_slope)

        return out * self.scale