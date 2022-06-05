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

@register("mlp")
class MLP(nn.Module):
    def __init__(self, in_features: int, 
                 hidden_features: int=None, 
                 out_features: int=None, 
                 act_layer: Callable=nn.GELU, 
                 drop: float=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x