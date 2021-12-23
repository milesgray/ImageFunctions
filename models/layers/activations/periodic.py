import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.layers import Scale
from .methods import *

from .registry import register

@register("sine")
class Sine(nn.Module):
    def __init__(self, scale: float=1.0, learnable: bool=False):
        super().__init__()
        self.scale = Scale(scale) if learnable else scale

    def forward(self, x: Tensor) -> Tensor:
        return x.sin()
    

@register("sines_cosines")
class SinesCosines(nn.Module):
    """
    Sines-cosines activation function
    It applies both sines and cosines and concatenates the results
    """
    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([x.sin(), x.cos()], dim=1)

