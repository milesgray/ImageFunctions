from argparse import Namespace
from typing import Tuple, Callable

import torch.nn as nn
import torch
from torch import Tensor

from .learnable import Scale, Balance

from .registry import register

@register("linear_residual")
class LinearResidual(nn.Module):
    def __init__(self, weight: float, 
                 learnable_weight: bool, 
                 weighting_type: str,
                 layer: Callable):
        super().__init__()
        
        assert weighting_type in ['shortcut', 'residual'], "weighting type must be one of 'shortcut' or 'residual'"

        self.weighting_type = weighting_type
        self.layer = layer
        self.scale = Scale(weight)
        self.balance = Balance()

    def forward(self, x: Tensor) -> Tensor:
        if self.weighting_type == 'shortcut':
            return self.balance(self.layer(x),
                                self.scale(x))
        elif self.weighting_type == 'residual':
            return self.balance(self.scale(self.layer(x)), x)
        else:
            raise ValueError