import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from argparse import Namespace
from .registry import register


@register("dcgan_disc")
class DCGANDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dis_out_real, dis_out_fake):
        device = dis_out_real.get_device()
        ones = torch.ones_like(dis_out_real, device=device, requires_grad=False)
        dis_loss = -torch.mean(nn.LogSigmoid()(dis_out_real) + nn.LogSigmoid()(ones - dis_out_fake))
        return dis_loss

@register("dcgan_gen")
class DCGANGeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dis_out_real, dis_out_fake):
        return -torch.mean(nn.LogSigmoid()(gen_out_fake))

@register("lsgan_disc")
class LSGANDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dis_out_real, dis_out_fake):
        dis_loss = 0.5*(dis_out_real - torch.ones_like(dis_out_real))**2 + 0.5*(dis_out_fake)**2
        return dis_loss.mean()

@register("lsgan_gen")
class LSGANGeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dis_out_real, dis_out_fake):
        dis_loss = 0.5*(dis_out_real - torch.ones_like(dis_out_real))**2 + 0.5*(dis_out_fake)**2
        return dis_loss.mean()

@register("gan_hinge_disc")
class HingeDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dis_out_real, dis_out_fake):
        return torch.mean(F.relu(1. - dis_out_real)) + torch.mean(F.relu(1. + dis_out_fake))

@register("gan_hinge_gen")
class HingeDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dis_out_real, gen_out_fake):
        return -torch.mean(gen_out_fake)

@register("gan_wgan_disc")
class WGANDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dis_out_real, dis_out_fake):
        return torch.mean(dis_out_fake - dis_out_real)

@register("gan_wgan_gen")
class WGANGeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dis_out_real, gen_out_fake):
        return -torch.mean(gen_out_fake)


@register("gan_grad")
class GradLoss(nn.Module):
    def __init__(self, style):
        super().__init__()
        self.style = style

    def forward(self, disc_model, real, fake=None):
        if self.style == "wgan":
            loss = grad_loss.compute_wgan_gp(disc_model, real, fake)
        elif self.style == "r1":
            loss = grad_loss.compute_r1_penalty(disc_model, real)
        return loss

@register("gan_nonsaturating_fake")
class FakeNonSaturatingLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, real, fake):
        return F.softplus(fake).mean()
@register("gan_nonsaturating_real")
class RealNonSaturatingLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, real, fake):
        return F.softplus(-real).mean()

@register("gan_hinge")
class HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, real, fake):
        real = F.normalize(real, dim=-1, p=2)
        fake = F.normalize(fake, dim=-1, p=2)
        return (F.relu(1 + real) + F.relu(1 - fake)).mean()

@register("gan_adversarial")
class AdversarialLoss(nn.Module):
    def __init__(self, gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0):
        """
        Args:
            gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
            real_label_val (float): The value for real label. Default: 1.0.
            fake_label_val (float): The value for fake label. Default: 0.0.
        """

        super().__init__()

        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError('GAN type %s is not implemented.' % self.gan_type)

    def _wgan_loss(self, x, target):
        """
        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.
        """
        return -x.mean() if target else x.mean()

    def _wgan_softplus_loss(self, x, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the ReLU function.
        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.
        Args:
            x (Tensor): Input tensor.
            target (bool): Target label.
        """
        return F.softplus(-x).mean() if target else F.softplus(x).mean()

    def get_target_label(self, x, target_is_real):
        """Get target label.
        Args:
            x (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.
        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise, return Tensor.
        """
        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return x.new_ones(x.size()) * target_val

    def forward(self, x, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(x, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:
                x = -x if target_is_real else x
                loss = self.loss(1 + x).mean()
            else:
                loss = -x.mean()
        else:  # other gan types
            loss = self.loss(x, target_label)

        return loss