import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .learnable import Scale
from .registry import register

@register("simam_attn")
class SimamAttention(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4, lean_gain=False):
        super().__init__()
        self.channels = channels
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
        if learn_gain:
            self.gain = Scale()
        else:
            self.gain = nn.Identity()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.gain(self.activaton(y))