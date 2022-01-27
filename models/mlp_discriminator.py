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

    def forward(self, x):
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

    return MLPDiscriminator(args)
