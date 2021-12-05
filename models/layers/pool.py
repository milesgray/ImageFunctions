import torch
import torch.nn as nn

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (
                torch.max(x,1)[0].unsqueeze(1),
                torch.mean(x,1).unsqueeze(1)
            ),
            dim=1
        )
