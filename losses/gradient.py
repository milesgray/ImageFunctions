import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, autograd

from argparse import Namespace
from .registry import register

def compute_gradient_penalty(type: str, discriminator, x_real, x_fake, y: Tensor=None):
    if type == "r1":
        return compute_r1_penalty(discriminator, x_real, y=y)
    elif type == "wgan-gp":
        return compute_wgan_gp(discriminator, x_real, x_fake, y=y)
    else:
        raise NotImplementedError(f'Unknown gradient penalty: {type}')

@register('r1_grad_penalty')
def compute_r1_penalty(discriminator, x_real, y: Tensor=None):
    """
    Computes gradient penalty using R1 regularization.
    Args:
    - y — class labels for cGAN
    """
    x_real = x_real.clone().requires_grad_(True) # So not to break the original data

    if y is None:
        outputs = discriminator(x_real)
    else:
        assert len(y) == len(x_real), f"Wrong shape: {y.shape} vs {x_real.shape}"
        outputs = discriminator(x_real, y)

    return compute_r1_penalty_from_outputs(outputs, x_real)


def compute_r1_penalty_from_outputs(d_outputs, x_real):
    """
    Computes R1 grad penalty based on the existing d_outputs to save memory
    """
    assert x_real.requires_grad

    grads = autograd.grad(
        d_outputs.sum(), x_real,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    assert grads.shape == x_real.shape, f"Wrong shape: {grads.shape}"

    r1_reg = grads.view(len(x_real), -1).pow(2).sum(dim=1)

    return r1_reg.mean()

@register('wgan_grad_penalty')
def compute_wgan_gp(discriminator, x_real, x_fake, y: Tensor=None):
    """
    Computes gradient penalty according to WGAN-GP paper
    Args:
    - y — class labels for cGAN
    """
    assert x_real.size() == x_fake.size(), f"Wrong shapes: {x_real.shape} vs {x_fake.shape}"

    shape = [x_real.size(0)] + [1] * (x_real.dim() - 1)
    alpha = torch.rand(shape).to(x_real.device)
    interpolations = x_real + alpha * (x_fake - x_real)

    interpolations = interpolations.to(x_real.device)
    interpolations.requires_grad_(True)

    if y is None:
        outputs = discriminator(interpolations)
    else:
        outputs = discriminator(interpolations, y)

    grads = autograd.grad(
        outputs,
        interpolations,
        grad_outputs=torch.ones(outputs.size()).to(interpolations.device),
        create_graph=True
    )[0].view(interpolations.size(0), -1)

    return ((grads.norm(p=2, dim=1) - 1) ** 2).mean()

@register("gradient")
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        filterx = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0. , 3.]])
        self.fx = filterx.expand(1,3,3,3).cuda()

        filtery = torch.tensor([[-3., -10, -3.], [0., 0., 0.], [3., 10. , 3.]])
        self.fy = filtery.expand(1,3,3,3).cuda()

    def forward(self, x, y):
        schxx = F.conv2d(x, self.fx, stride=1, padding=1)
        schxy = F.conv2d(x, self.fy, stride=1, padding=1)
        gradx = torch.sqrt(torch.pow(schxx, 2) + torch.pow(schxy, 2) + 1e-6)
        
        schyx = F.conv2d(y, self.fx, stride=1, padding=1)
        schyy = F.conv2d(y, self.fy, stride=1, padding=1)
        grady = torch.sqrt(torch.pow(schyx, 2) + torch.pow(schyy, 2) + 1e-6)
        
        loss = F.l1_loss(gradx, grady)
        return loss