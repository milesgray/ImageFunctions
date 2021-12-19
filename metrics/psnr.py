import torch
import numpy as np

def calc_psnr(sr, hr, scale=1, rgb_range=1, channels=3):
    diff = (sr - hr) / rgb_range
    valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)

def psnr(p0, p1, peak=255.):
    return 10*torch.log10(peak**2/torch.mean((1.*p0-1.*p1)**2))