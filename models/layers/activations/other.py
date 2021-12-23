import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.layers import Scale, Balance
from .methods import mish, logcosh

from .registry import register



@register("logcosh")
class LogCosh(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return logcosh(x)
