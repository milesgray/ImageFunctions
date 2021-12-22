import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .methods import *

from .registry import register

@register("xsigmoid")
class XSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return xsigmoid(x)


@register("centered_sigmoid")
class CenteredSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return centeredsigmoid(x)


@register("unit_centered_sigmoid")
class UnitCenteredSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return unitcenteredsigmoid(x)