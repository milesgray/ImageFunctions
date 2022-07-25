import torch
import torch.nn as nn

class SpatialSoftmax2d(nn.Module):
    def __init__(self, temp: float=1.0, requires_grad: bool=True):
        super().__init__()
        self.temp = nn.Parameter(torch.FloatTensor([temp]), requires_grad=requires_grad)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        x = self.softmax(x)
        return x * self.temp.to(x.device)

class ChannelSoftmax2d(nn.Module):
    def __init__(self, temp: float=1.0, requires_grad: bool=True):
        super().__init__()
        self.temp = nn.Parameter(torch.FloatTensor([temp]), requires_grad=requires_grad)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.softmax(x)
        return x * self.temp.to(x.device)
    
    
class ChannelGumbelMax2d(nn.Module):
    def __init__(self, tau: float=1.0, hard: bool=False, dim: int=-1, requires_grad: bool=True):
        super().__init__()
        self.tau = tau
        self.hard = hard
        self.dim = dim
        self.temp = nn.Parameter(torch.FloatTensor([temp]), requires_grad=requires_grad)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.gumbel_softmax(x, tau=self.tau, hard=self.hard, dim=self.dim)
        return x * self.temp.to(x.device)