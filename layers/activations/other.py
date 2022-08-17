import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .methods import logcosh, gaussian, hat

from .registry import register

@register("logcosh")
class LogCosh(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return logcosh(x)

@register("gaussian")
class Gaussian(torch.nn.Module):
    def __init__(self, a: int):
        super().__init__()
        self.a = a

    def forward(self, x: Tensor) -> Tensor:
        return gaussian(x, a)

@register("hat")
class Hat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return hat(x)