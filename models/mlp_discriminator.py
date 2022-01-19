import math
from argparse import Namespace
from typing import Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

import models
import models.layers as layers
import models.layers.activations as activations
from models import register
from utility import make_coord

def create_activate(name, **kwargs):
    return activations.make({"name": name, "args": kwargs})

def sn_wrapper(module: nn.Module, use_sn: bool, *sn_args, **sn_kwargs) -> nn.Module:
    """
    So not to wrap it everywhere
    """
    if use_sn:
        return nn.utils.spectral_norm(module, *sn_args, **sn_kwargs)
    else:
        return module
class FourierINR(nn.Module):
    """
    INR with Fourier features as specified in https://people.eecs.berkeley.edu/~bmild/fourfeat/
    """
    def __init__(self, in_features, args: Namespace, num_fourier_feats=64, layer_sizes=[64,64,64], out_features=64, 
                 has_bias=True, activation="leaky_relu", 
                 learnable_basis=True,):
        super().__init__()

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
def make_mlp_disc(in_dim=128, 
                  hidden_list=[128,128], 
                  out_dim=1, 
                  activation="scaled_leaky_relu", 
                  has_bias=True):
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
