import torch
import torch.nn as nn

from .registry import register

@register("zpool")
class ZPool(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(
            (
                torch.max(x, self.dim)[0].unsqueeze(1),
                torch.mean(x, self.dim).unsqueeze(1)
            ),
            dim=1
        )
