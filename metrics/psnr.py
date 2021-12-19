import torch
import torch.nn as nn
import numpy as np
from .registry import register

def calc_psnr(sr, hr, scale=1, rgb_range=1, channels=3):
    diff = (sr - hr) / rgb_range
    valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)

def psnr(p0, p1, peak=255.):
    return 10*torch.log10(peak**2/torch.mean((1.*p0-1.*p1)**2))

@register("psnr")
class PSNRMetric(nn.Module):
    def __init__(self, scale=1, rgb_range=1, channels=3):
        super().__init__()
        self.scale = scale
        self.rgb_range = rgb_range
        self.channels = channels
        
    def forward(self, sr, hr):
        result = calc_psnr(sr, hr, 
                           scale=self.scale, 
                           rgb_range=self.rgb_range,
                           channels=self.channels)