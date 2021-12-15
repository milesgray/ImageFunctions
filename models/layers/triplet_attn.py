import torch
import torch.nn as nn

from .gate import PoolGate

from .registry import register

@register("triplet_attention")
class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False, kernel_size=7):
        super().__init__()
        self.cw = PoolGate(2, 1, kernel_size=kernel_size)
        self.hc = PoolGate(2, 1, kernel_size=kernel_size)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = PoolGate(2, 1, kernel_size=kernel_size)

    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out