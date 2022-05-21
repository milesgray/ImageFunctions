import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .methods import xtanh, unitcenteredtanh, mish

from .registry import register


@register("mish2")
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

    def forward(self, x):
        """
        Forward pass of the function.
        """
        return mish(x)

@register("xtanh")
class XTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return xtanh(x)

@register("centered_tanh")
class UnitCenteredTanh(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return unitcenteredtanh(x)
