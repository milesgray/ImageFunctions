import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.attn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size, 
                      padding=padding, 
                      bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.attn(max_out)

class CatPoolSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, 
                      padding=padding, 
                      bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.attn(x)