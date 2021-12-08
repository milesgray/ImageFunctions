from torch import nn
from .learnable import Balance
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.attn = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.PReLU(channel // reduction),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.attn(y)
        return x * y

class MixPoolChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.net = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.PReLU(channel // reduction),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias)
        )
        self.balance = Balance(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.net(self.avg_pool(x))
        max_out = self.net(self.max_pool(x))
        out = self.balance(avg_out, max_out)
        return self.sigmoid(out)