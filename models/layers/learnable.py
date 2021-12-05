import torch
from torch import nn

class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]), requires_grad=True)

    def forward(self, x):
        return x * self.scale

class Balance(nn.Module):
    def __init__(self, init_value=0.5):
        super().__init__()
        self.beta = nn.Parameter(torch.FloatTensor([init_value]), requires_grad=True)

    def forward(self, x, y):
        return (x * self.beta) + (y * (1 - self.beta))
