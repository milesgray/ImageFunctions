from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register

@register("conditional_bn_2d")
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = nn.Embedding(num_classes, num_features)
        self.beta = nn.Embedding(num_classes, num_features)
        torch.nn.init.ones_(self.gamma.weight)
        torch.nn.init.zeros_(self.beta.weight)

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma(y)
        beta = self.beta(y)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out
    
@register("layer_norm")
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

@register("spectral_norm")
class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
    
@register("filter_response_norm")
class FilterResponseNorm(nn.Module):
    """ Filter Response Normalization """
    def __init__(self, num_features, ndim, eps=None, learnable_eps=False):
        """
        Args:
            num_features
            ndim
            eps: if None is given, use the paper value as default.
                from paper, fixed_eps=1e-6 and learnable_eps_init=1e-4.
            learnable_eps: turn eps to learnable parameter, which is recommended on
                fully-connected or 1x1 activation map.
        """
        super().__init__()
        if eps is None:
            if learnable_eps:
                eps = 1e-4
            else:
                eps = 1e-6

        self.num_features = num_features
        self.init_eps = eps
        self.learnable_eps = learnable_eps
        self.ndim = ndim

        self.mean_dims = list(range(2, 2+ndim))

        self.weight = nn.Parameter(torch.ones([1, num_features] + [1]*ndim))
        self.bias = nn.Parameter(torch.zeros([1, num_features] + [1]*ndim))
        if learnable_eps:
            self.eps = nn.Parameter(torch.as_tensor(eps))
        else:
            self.register_buffer('eps', torch.as_tensor(eps))

    def forward(self, x):
        # normalize
        nu2 = x.pow(2).mean(self.mean_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # modulation
        x = x * self.weight + self.bias

        return x

    def extra_repr(self):
        return 'num_features={}, init_eps={}, ndim={}'.format(
                self.num_features, self.init_eps, self.ndim)

FilterResponseNorm1d = partial(FilterResponseNorm, ndim=1, learnable_eps=True)
FilterResponseNorm2d = partial(FilterResponseNorm, ndim=2)

@register("ada_in")
class AdaIN(nn.Module):
    """
    Adaptive Instance normalization.
    reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
    """
    def __init__(self, n_channels, code):
        super().__init__()
        
        self.norm = nn.InstanceNorm2d(n_channels, affine=False, eps=1e-8)
        self.A = ScaledLinear(code, n_channels * 2)
        
        # StyleGAN
        # self.A.linear.bias.data = torch.cat([torch.ones(n_channels), torch.zeros(n_channels)])
        
    def forward(self, x, style):
        """
        x - (N x C x H x W)
        style - (N x (Cx2))
        """        
        # Project project style vector(w) to  mu, sigma and reshape it 2D->4D to allow channel-wise operations        
        style = self.A(style)
        y = style.view(style.shape[0], 2, style.shape[1]//2).unsqueeze(3).unsqueeze(4)

        x = self.norm(x)
        
        return torch.addcmul(y[:, 1], value=1., tensor1=y[:, 0] + 1, tensor2 = x)        

@register("ada_pn")
class AdaPN(nn.Module):
    """
    Pixelwise feature vector normalization.
    reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
    """
    def __init__(self, n_channels, code):
        super().__init__()
        self.A = ScaledLinear(code, n_channels * 2)

    def forward(self, x, style, alpha=1e-8):
        """
        x - (N x C x H x W)
        style - (N x (Cx2))
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        # Project project style vector(w) to  mu, sigma and reshape it 2D->4D to allow channel-wise operations  
        style = self.A(style)
        z = style.view(style.shape[0], 2, style.shape[1]//2).unsqueeze(3).unsqueeze(4)
        # original PixelNorm
        y = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)  # [N1HW]
        y = x / y  # normalize the input x volume        
        # addcmul like in AdaIN
        return torch.addcmul(z[:, 1], value=1., tensor1=z[:, 0] + 1, tensor2=y)

@register("stddev_batch")
class MinibatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer
    reference: https://github.com/akanimax/pro_gan_pytorch/blob/master/pro_gan_pytorch/CustomLayers.py#L300
    """

    def __init__(self):
        """
        derived class constructor
        """
        super().__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape

        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)

        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y