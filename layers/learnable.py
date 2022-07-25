import torch
from torch import nn

from .registry import register

@register("scale")
class Scale(nn.Module):
    def __init__(self, init_value=1e-3, requires_grad=True):
        super().__init__()
        self.scale = nn.Parameter(
            torch.FloatTensor([init_value]), 
            requires_grad=requires_grad)

    def forward(self, x):
        return x * self.scale

@register("balance")
class Balance(nn.Module):
    def __init__(self, init_value=0.5, requires_grad=True):
        super().__init__()
        self.beta = nn.Parameter(
            torch.FloatTensor([init_value]), 
            requires_grad=requires_grad)

    def forward(self, x, y):
        return (x * self.beta) + (y * (1 - self.beta))
