import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.layers import Scale
from .methods import *

from .registry import register

@register("scaled_leaky_relu")
class ScaledLeakyReLU(nn.Module):
    def __init__(self, 
                 negative_slope: float=0.2, 
                 scale: float=None,
                 learnable: bool=False):
        super().__init__()        
        self.learnable = learnable
        self.negative_slope = nn.Parameter(torch.FloatTensor([negative_slope]), 
                                           requires_grad=self.learnable)
        
        scale = math.sqrt(2 / (1 + negative_slope ** 2)) if scale is None else scale
        self.scale = Scale(scale)

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return self.scale(out)
    
@register("adaptive_leaky_relu")
class AdaptiveLeakyReLU(nn.Module):
    def __init__(self, negative_slope: float=0.2, scale: float=None):
        super().__init__()

        self.negative_slope = nn.Parameter(torch.FloatTensor([negative_slope]), requires_grad=True)
        scale = math.sqrt(2 / (1 + negative_slope ** 2)) if scale is None else scale
        self.scale = Scale(scale)

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=self.negative_slope)

        return self.scale(out)