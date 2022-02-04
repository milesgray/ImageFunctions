import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .methods import logcosh

from .registry import register

@register("logcosh")
class LogCosh(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return logcosh(x)
