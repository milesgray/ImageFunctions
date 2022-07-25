import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .learnable import Scale, Balance
from .registry import register

@register("simam_attn")
class SimamAttention(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4, learn_gain=False, apply_attn=False):
        super().__init__()
        self.channels = channels
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
        if learn_gain:
            self.gain = Scale()
        else:
            self.gain = nn.Identity()
        self.apply_attn = apply_attn


    def forward(self, x):
        n = x.shape[2] * x.shape[3] - 1
        # square of (t - u)
        d = (x - x.mean(dim=[2,3])).pow(2)
        # d.sum() / n is channel variance
        v = d.sum(dim=[2,3]) / n
        # E_inv groups all importance of X
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5        

        y = x * self.gain(self.activaton(e_inv))

        if self.apply_attn:
            return x * y
        else:
            return y