import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ImageFunctions.layers import Scale
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

@register("tlu")
class TLU(nn.Module):
    """ Thresholded Linear Unit """
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1),
                                requires_grad=True)

    def forward(self, x):
        return torch.max(x, self.tau)