import torch
import torch.nn as nn
from torch.autograd import Function
from torch import Tensor

from .registry import register

@register("shuffle_down")
class ShuffleDown(nn.Module):
    """ https://github.com/dariofuoli/RLSP/blob/master/pytorch/functions.py
    Originally from RLSP video SR project """
    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor
        
    def forward(self, x):
        # format: (B, C, H, W)
        b, c, h, w = x.shape

        assert h % self.factor == 0 and w % self.factor == 0, "H and W must be a multiple of " + str(self.factor) + "!"

        n = x.reshape(b, c, int(h/self.factor), self.factor, int(w/self.factor), self.factor)
        n = n.permute(0, 3, 5, 1, 2, 4)
        n = n.reshape(b, c*self.factor**2, int(h/self.factor), int(w/self.factor))

        return n


@register("shuffle_up")
class ShuffleUp(nn.Module):
    """ https://github.com/dariofuoli/RLSP/blob/master/pytorch/functions.py
    Originally from RLSP video SR project """
    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor
        
    def forward(self, x):
        # format: (B, C, H, W)
        b, c, h, w = x.shape

        assert c % self.factor**2 == 0, "C must be a multiple of " + str(self.factor**2) + "!"

        n = x.reshape(b, self.factor, self.factor, int(c/(self.factor**2)), h, w)
        n = n.permute(0, 3, 4, 1, 5, 2)
        n = n.reshape(b, int(c/(self.factor**2)), self.factor*h, self.factor*w)

        return n