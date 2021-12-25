import torch
from torch import nn

from .registry import register

@register("space_to_depth")
class SpaceToDepth(nn.Module):
    def __init__(self, bs):
        super().__init__()
        self.bs = bs

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)
        return x

@register("gaussian_0d")
class GaussianTransform0d(nn.Module):
    def __init__(self, scale=512, requires_grad=True):
        """ scale matches input and does not change shape, only values
        """
        super().__init__()

        self.weight = nn.Parameter(torch.ones(scale), 
                                   requires_grad=requires_grad)
        self.bias = nn.Parameter(torch.zeros(scale), 
                                 requires_grad=requires_grad)

    def forward(self, x):
        z = self.weight * x
        x = self.bias.exp() * x
        z = z + (x - 1)
        return z
@register("gaussian_1d")
class GaussianTransform1d(nn.Module):
    def __init__(self, scale=512):
        """ scale matches input and does not change shape, only values
        """
        super().__init__()

        if isinstance(scale, int):
            scale = (scale, scale)

        self.A = nn.Linear(scale[0], scale[1], bias=True)

    def forward(self, x):
        z = self.A.weight * x
        x = self.A.bias.exp() * x

        z = z + (x - 1)
        return z
@register("gaussian_2d")
class GaussianTransform2d(nn.Module):
    def __init__(self, scale=(4,4), requires_grad=True):
        super().__init__()
        if isinstance(scale, int):
            scale = (scale, scale)
        self.weight = nn.Parameter(torch.zeros(1, scale[0], scale[1], 1), requires_grad=requires_grad)
        self.bias = nn.Parameter(torch.zeros(1, scale[0], scale[1], 1), requires_grad=requires_grad)

    def forward(self, x):
        x = self.weight * x
        z = self.bias.exp() * x
        return x + (z - 1)
# ------------------------------------------------------------------------------------------------------------------
#  Affine Gaussian-ish Transformation
# Used for fine grain corrective projection after a heavier transform
# ------------------------------------------------------------------------------------------------------------------
class AffineTransform0d(nn.Module):
    def __init__(self, scale=512, requires_grad=True):
        """ scale matches input and does not change shape, only values
        """
        super().__init__()

        #self.A = ScaledLinear(scale[0], scale[1], bias=True)
        self.weight = nn.Parameter(torch.zeros(1, scale), requires_grad=requires_grad)
        self.bias = nn.Parameter(torch.zeros(scale), requires_grad=requires_grad)

    def forward(self, x):
        #z = self.A.linear.__dict__['_parameters']['weight_orig'] * x
        #x = self.A.linear.bias.exp() * x
        z = self.weight * x
        x = self.bias.exp() * x
        z = z + x
        return z
class AffineTransform1d(nn.Module):
    def __init__(self, scale=512, requires_grad=True):
        """ scale matches input and does not change shape, only values
        """
        super().__init__()

        if isinstance(scale, int):
            scale = (scale, scale)

        #self.A = ScaledLinear(scale[0], scale[1], bias=True)
        self.weight = nn.Parameter(torch.zeros(1, 1, scale[1]), requires_grad=requires_grad)
        self.bias = nn.Parameter(torch.zeros(scale[1]), requires_grad=requires_grad)

    def forward(self, x):
        z = self.weight * x
        x = self.bias.exp() * x

        z = z + x
        return z
class AffineTransform2d(nn.Module):
    def __init__(self, scale=(4,4), requires_grad=True):
        super().__init__()
        if isinstance(scale, int):
            scale = (scale, scale)
        self.weight = nn.Parameter(torch.zeros(1, scale[0], scale[1], 1), requires_grad=requires_grad)
        self.biass = nn.Parameter(torch.zeros(1, scale[0], scale[1], 1), requires_grad=requires_grad)

    def forward(self, x):
        x = self.weight.exp() * x
        z = self.bias + x
        return z