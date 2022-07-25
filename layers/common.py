import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .registry import register
from .registry import create as create_layer
from .learnable import Balance, Scale
    
@register("residual")
class Residual(nn.Module):
    def __init__(self, fn, balance=False, scale=False):
        super().__init__()
        self.fn = fn
        self.fuse = Balance() if balance else torch.add
        self.scale = Scale() if scale else nn.Identity()
        
    def forward(self, x, **kwargs):
        return self.fuse(self.scale(self.fn(x, **kwargs)), x)

@register("prenorm")
class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm=nn.LayerNorm):
        super().__init__()
        norm = create(norm)
        self.norm = norm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

@register("prenorm_residual")
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn, norm=nn.LayerNorm):
        super().__init__()
        norm = create(norm)
        self.norm = norm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

@register("basic_conv")
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, 
                 stride=1, 
                 dilation=1, 
                 groups=1, 
                 act=nn.ReLU(True), 
                 conv=nn.Conv2d, 
                 bn=True, 
                 bias=False):
        super().__init__()
        self.out_channels = out_planes
        
        padding = int((kernel_size - 1) / 2) * dilation

        m = [conv(in_planes, out_planes, 
                  kernel_size=kernel_size, 
                  stride=stride, 
                  padding=padding, 
                  dilation=dilation, 
                  groups=groups, 
                  bias=bias)]
        if bn:
            m.append( nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) )
        if act is not None:
            m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)
    
@register("basic_block")
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, 
                 bias=False,
                 bn=True, 
                 conv=nn.Conv2d,
                 act=nn.ReLU(True)):
        super().__init__()
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

@register("upsampler")
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super().__init__(*m)

@register("flatten")
class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)