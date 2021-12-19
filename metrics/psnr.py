import torch

def calc_psnr(sr, hr, scale=1, rgb_range=1, channels=3):
    diff = (sr - hr) / rgb_range
    valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)
