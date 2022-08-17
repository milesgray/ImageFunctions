import torch
import torch.nn as nn
from torch.nn import functional as F

from .registry import register
from .learnable import Balance, Scale

@register("non_local_attn")
class NonLocalAttention(nn.Module):
    def __init__(self, channels: int, conv: nn.Module=nn.Conv2d):
        """A non-local block as used in SA-GAN.

        Args:
            ch (int): Number of channels/filters
            which_conv (nn.Module, optional): A 2D Convolutional module to use. Defaults to SNConv2d.
        """
        super().__init__()
        # Channel multiplier
        self.channels = channels
        self.conv = conv
        self.theta = self.conv(
            self.channels, self.channels // 8, 
            kernel_size=1, 
            padding=0, 
            bias=False)
        self.phi = self.conv(
            self.channels, self.channels // 8, 
            kernel_size=1, 
            padding=0, 
            bias=False)
        self.g = self.conv(
            self.channels, self.channels // 2, 
            kernel_size=1, 
            padding=0, 
            bias=False)
        self.o = self.conv(
            self.channels // 2, 
            self.channels, 
            kernel_size=1, 
            padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = Scale(0.0)
        self.residual_balance = Balance(0.0)

    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.channels // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.channels // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.channels // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1,
                                                           self.channels // 2,
                                                           x.shape[2],
                                                           x.shape[3]))
        return self.residual_balance(self.gamma(o), x)