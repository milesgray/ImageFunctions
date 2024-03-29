import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .registry import register

@register('split_bn')
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features: int, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                x.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                x, self.running_mean, self.running_var, 
                self.weight, self.bias, False, self.momentum, self.eps)
