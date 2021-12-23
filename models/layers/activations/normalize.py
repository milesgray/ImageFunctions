import torch
import torch.nn as nn
from torch import Tensor

from .methods import normalizer

from .registry import register


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