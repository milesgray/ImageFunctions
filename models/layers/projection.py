import torch
from torch import nn
from torch.autograd import Function

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
@register("affine_0d")
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
@register("affine_1d")
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
@register("affine_2d")
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

@register("ball_project")
class BallProjection(nn.Module):
    """
    Constraint norm of an input noise vector to be sqrt(latent_code_size)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div((torch.mean(x.pow(2), dim=1, keepdim=True).add(1e-8)).pow(0.5))

class IntermediateNoise(nn.Module):
    def __init__(self, inp_c):
        """Normal Distribution with Learnable Scale

        Args:
            inp_c (int): Number of channels in dim 1 of weights
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, inp_c, 1, 1), requires_grad=True)
        self.noise = None
    
    def forward(self, x, noise=None):
        if self.training:
            if noise is None and self.noise is None:
                noise = torch.randn(x.shape[0], 1, x.shape[-2], x.shape[-1]).to(x.device)
            elif noise is None:
                noise = self.noise
            return x + (noise * self.weight)
        else:
            return x
class BlurFunctionBackward(Function):
    """
    Official Blur implementation
    https://github.com/adambielski/perturbed-seg/blob/master/stylegan.py
    """
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None

class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None

blur = BlurFunction.apply

@register("blur")
class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)

# ------------------------------------------------------------------------------------------------------------------
# Blur from original ALAE
# https://github.com/podgorskiy/ALAE/blob/master/net.py#L49
# ------------------------------------------------------------------------------------------------------------------
@register("blur_simple")
class BlurSimple(nn.Module):
    def __init__(self, channels):
        super().__init__()
        f = np.array([1, 2, 1], dtype=np.float32)
        f = f[:, np.newaxis] * f[np.newaxis, :]
        f /= np.sum(f)
        kernel = torch.Tensor(f).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel)
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, groups=self.groups, padding=1)