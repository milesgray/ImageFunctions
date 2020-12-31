import math
from argparse import Namespace
from typing import Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

import models
from models import register
from utils import make_coord

def sn_wrapper(module: nn.Module, use_sn: bool, *sn_args, **sn_kwargs) -> nn.Module:
    """
    So not to wrap it everywhere
    """
    if use_sn:
        return nn.utils.spectral_norm(module, *sn_args, **sn_kwargs)
    else:
        return module

class LinearResidual(nn.Module):
    def __init__(self, args: Namespace, transform: Callable):
        super().__init__()

        self.args = args
        self.transform = transform
        self.weight = nn.Parameter(
            torch.tensor(args.weight).float(), requires_grad=args.learnable_weight)

    def forward(self, x: Tensor) -> Tensor:
        if self.args.weighting_type == 'shortcut':
            return self.transform(x) + self.weight * x
        elif self.args.weighting_type == 'residual':
            return self.weight * self.transform(x) + x
        else:
            raise ValueError

def create_activation(activation_type: str, *args, **kwargs) -> nn.Module:
    if activation_type == 'leaky_relu':
        return nn.LeakyReLU(*args, **kwargs)
    elif activation_type == 'scaled_leaky_relu':
        return ScaledLeakyReLU(*args, **kwargs)
    elif activation_type == 'relu':
        return nn.ReLU(*args, **kwargs)
    elif activation_type == 'tanh':
        return nn.Tanh(*args, **kwargs)
    elif activation_type == 'none' or activation_type is None:
        return nn.Identity(*args, **kwargs)
    elif activation_type == 'centered_sigmoid':
        return CenteredSigmoid(*args, **kwargs)
    elif activation_type == 'sine':
        return Sine(*args, **kwargs)
    elif activation_type == 'unit_centered_sigmoid':
        return UnitCenteredSigmoid(*args, **kwargs)
    elif activation_type == 'unit_centered_tanh':
        return UnitCenteredTanh(*args, **kwargs)
    elif activation_type == 'sines_cosines':
        return SinesCosines(*args, **kwargs)
    elif activation_type == 'normalizer':
        return Normalizer(*args, **kwargs)
    elif activation_type == 'normal_clip':
        return NormalClip(*args, **kwargs)
    else:
        raise NotImplementedError(f'Unknown activation type: {activation_type}')


class CenteredSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid() * 2 - 1


class UnitCenteredSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid() * 2


class UnitCenteredTanh(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh() + 1


class Sine(nn.Module):
    def __init__(self, scale: float=1.0):
        super(Sine, self).__init__()
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        return x.sin()


class Normalizer(nn.Module):
    """
    Just normalizes its input
    """
    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2, f"Wrong shape: {x.shape}"

        return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)


class NormalClip(nn.Module):
    """
    Clips input values into [-2, 2] region
    """
    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x, -2, 2)


class SinesCosines(nn.Module):
    """
    Sines-cosines activation function
    It applies both sines and cosines and concatenates the results
    """
    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([x.sin(), x.cos()], dim=1)

class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope: float=0.2, scale: float=None):
        super().__init__()

        self.negative_slope = negative_slope
        self.scale = math.sqrt(2 / (1 + negative_slope ** 2)) if scale is None else scale

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * self.scale
        
class FourierINR(nn.Module):
    """
    INR with Fourier features as specified in https://people.eecs.berkeley.edu/~bmild/fourfeat/
    """
    def __init__(self, in_features, args: Namespace, num_fourier_feats=64, layer_sizes=[64,64,64], out_features=64, 
                 has_bias=True, activation="leaky_relu", 
                 learnable_basis=True,):
        super(FourierINR, self).__init__()

        layers = [
            nn.Linear(num_fourier_feats * 2, layer_sizes[0], bias=has_bias),
            create_activation(activation)
        ]

        for index in range(len(layer_sizes) - 1):
            transform = nn.Sequential(
                nn.Linear(layer_sizes[index], layer_sizes[index + 1], bias=has_bias),
                create_activation(activation)
            )

            if args.residual.enabled:
                layers.append(LinearResidual(args.residual, transform))
            else:
                layers.append(transform)

        layers.append(nn.Linear(layer_sizes[-1], out_features, bias=has_bias))

        self.model = nn.Sequential(*layers)

        # Initializing the basis
        basis_matrix = args.scale * torch.randn(num_fourier_feats, in_features)
        self.basis_matrix = nn.Parameter(basis_matrix, requires_grad=learnable_basis)

    def compute_fourier_feats(self, coords: Tensor) -> Tensor:
        sines = (2 * np.pi * coords @ self.basis_matrix.t()).sin() # [batch_size, num_fourier_feats]
        cosines = (2 * np.pi * coords @ self.basis_matrix.t()).cos() # [batch_size, num_fourier_feats]

        return torch.cat([sines, cosines], dim=1) # [batch_size, num_fourier_feats * 2]

    def forward(self, coords: Tensor) -> Tensor:
        return self.model(self.compute_fourier_feats(coords))

class MLPDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()

        fourier_args = Namespace()
        fourier_args.scale = 1.0
        fourier_args.residual = Namespace()
        fourier_args.residual.weight = 1.0
        fourier_args.residual.weighting_type = 'residual'
        fourier_args.residual.learnable_weight = True
        fourier_args.residual.enabled = True

        self.fourier = FourierINR(2, fourier_args, num_fourier_feats=args.in_dim, out_features=args.in_dim)

        layers = []
        lastv = args.in_dim
        for hidden in args.hidden_list[1:]:
            layers.append(sn_wrapper(nn.Linear(lastv, hidden), True))
            transform = nn.Sequential(
                sn_wrapper(nn.Linear(hidden, hidden, bias=args.has_bias), True),
                create_activation(args.activation)
            )
            layers.append(LinearResidual(args.residual, transform))
            lastv = hidden
        layers.append(nn.Linear(lastv, args.out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, coord=None):
        if coord is not None:
            x = self.fourier(x)
        x = x.view(x.shape[0], -1)
        x = self.layers(x)
        return x



@register('mlp_disc')
def make_mlp_disc(in_dim=128, hidden_list=[128,128], out_dim=1, activation="scaled_leaky_relu", has_bias=True):
    args = Namespace()
    args.in_dim = in_dim
    args.hidden_list = hidden_list
    args.out_dim = out_dim
    args.activation = activation
    args.has_bias = has_bias
    args.residual = Namespace()
    args.residual.weight = 1.0
    args.residual.weighting_type = 'residual'
    args.residual.learnable_weight = True
    args.residual.enabled = True

    return MLPDiscriminator(args)
