import torch
import torch.nn as nn
from torch import Tensor

from .registry import register



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
        super(Sine, self).__init__()
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
     