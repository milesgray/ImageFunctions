from functools import partial
import torch
import torch.nn as nn

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
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
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