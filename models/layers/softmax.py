import torch
import torch.nn as nn
from .learnable import Scale

class SpatialSoftmax2d(nn.Module):
    def __init__(self, temp=1.0, requires_grad=True):
        super().__init__()
        self.scale_temp = Scale(temp)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        x = self.softmax(x)
        return self.scale_temp(x)

class ChannelSoftmax2d(nn.Module):
    def __init__(self, temp=1.0, requires_grad=True):
        super().__init__()
        self.temp = Scale(temp)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.softmax(x)
        return x * self.temp