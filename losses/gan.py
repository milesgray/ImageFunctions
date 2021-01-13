import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from argparse import Namespace
from .registry import register


@register("dcgan_disc_loss")
class DCGANDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dis_out_real, dis_out_fake):
        device = dis_out_real.get_device()
        ones = torch.ones_like(dis_out_real, device=device, requires_grad=False)
        dis_loss = -torch.mean(nn.LogSigmoid()(dis_out_real) + nn.LogSigmoid()(ones - dis_out_fake))
        return dis_loss

@register("dcgan_gen_loss")
class DCGANGeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dis_out_fake):
        return -torch.mean(nn.LogSigmoid()(gen_out_fake))

@register("lsgan_disc_loss")
class LSGANDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dis_out_real, dis_out_fake):
        dis_loss = 0.5*(dis_out_real - torch.ones_like(dis_out_real))**2 + 0.5*(dis_out_fake)**2
        return dis_loss.mean()

@register("lsgan_gen_loss")
class LSGANGeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dis_out_real, dis_out_fake):
        dis_loss = 0.5*(dis_out_real - torch.ones_like(dis_out_real))**2 + 0.5*(dis_out_fake)**2
        return dis_loss.mean()

@register("hinge_disc_loss")
class HingeDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dis_out_real, dis_out_fake):
        return torch.mean(F.relu(1. - dis_out_real)) + torch.mean(F.relu(1. + dis_out_fake))

@register("hinge_gen_loss")
class HingeDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_out_fake):
        return -torch.mean(gen_out_fake)

@register("wgan_disc_loss")
class WGANDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dis_out_real, dis_out_fake):
        return torch.mean(dis_out_fake - dis_out_real)

@register("wgan_gen_loss")
class WGANGeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_out_fake):
        return -torch.mean(gen_out_fake)
