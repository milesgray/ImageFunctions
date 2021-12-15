import torch
import torch.nn as nn

class SpatialSoftmax2d(nn.Module):
    def __init__(self, temp=1.0, requires_grad=True):
        super().__init__()
        self.temp = nn.Parameter(torch.FloatTensor([temp]), requires_grad=requires_grad)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        x = self.softmax(x)
        return x * self.temp

class ChannelSoftmax2d(nn.Module):
    def __init__(self, temp=1.0, requires_grad=True):
        super().__init__()
        self.temp = nn.Parameter(torch.FloatTensor([temp]), requires_grad=requires_grad)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.softmax(x)
        return x * self.temp