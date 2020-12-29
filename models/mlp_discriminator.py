from argparse import Namespace

import torch.nn as nn

from models import register

class MLPDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        layers = []
        lastv = args.in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x



@register('mlp_disc')
def make_mlp_disc(in_dim=128, hidden_list=[128,128], out_dim=1):
    args = Namespace()
    args.in_dim = in_dim
    args.hidden_list = hidden_list
    args.out_dim = out_dim

    return MLPDiscriminator(args)
