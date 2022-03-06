import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from argparse import Namespace
from einops import rearrange

from .registry import register

from .activations import create as create_activation
from .linear_residual import LinearResidual

@register("lft_position")
class LFTPositionEncoding(nn.Module):
    def __init__(self, temperature, token_dim):
        super().__init__()
        self.temperature = temperature
        self.token_dim = token_dim

    def make_grid(self, token_dim, temperature=self.temperature):
        grid_dim = torch.linspace(0, token_dim - 1, token_dim, dtype=torch.float32)
        grid_dim = 2 * (grid_dim // 2) / token_dim
        grid_dim = temperature ** grid_dim
        return grid_dim

    def forward(self, x, dim: list, token_dim):
        assert len(x.size()) == 5, 'the object of position encoding requires 5-dim tensor! '
        grid_dim = self.make_grid(token_dim, temperature=self.temperature)
        position = None
        for index in range(len(dim)):
            pos_size = [1, 1, 1, 1, self.token_dim]
            length = x.size(dim[index])
            pos_size[dim[index]] = length

            pos_dim = (torch.linspace(0, length - 1, length, dtype=torch.float32).view(-1, 1) / grid_dim).to(x.device)
            pos_dim = torch.cat([pos_dim[:, 0::2].sin(), pos_dim[:, 1::2].cos()], dim=1)
            pos_dim = pos_dim.view(pos_size)

            if position is None:
                position = pos_dim
            else:
                position = position + pos_dim
            pass

        position = rearrange(position, 'b 1 a h w dim -> b dim a h w')

        return position / len(dim)


@register("fourier_position")
class FourierPositionEncoding(nn.Module):
    """
    INR with Fourier features as specified in https://people.eecs.berkeley.edu/~bmild/fourfeat/
    """
    def __init__(self, in_features, args, 
                 num_fourier_feats=64, 
                 layer_sizes=[64,64,64], 
                 out_features=64, 
                 has_bias=True, 
                 activation="leaky_relu", 
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
                layers.append(LinearResidual(transform, **args.residual))
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