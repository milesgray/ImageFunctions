import torch
import torch.nn as nn

from .registry import register

@register("stats")
class StatsLoss(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def calc_stats(self, feat, dim=1, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        N, C = size[:2]
        feat_var = feat.var(dim=dim) + eps
        feat_std = feat_var.sqrt()
        feat_mean = feat.mean(dim=dim)
        return {
            "mean": feat_mean, 
            "std": feat_std, 
            "var": feat_var, 
            "max": feat.max(), 
            "min": feat.min()
        }
        
    def forward(self, x, y):
        x_stats = self.calc_mean_std(x, dim=self.dim)
        y_stats = self.calc_mean_std(y, dim=self.dim)

        mean_loss = F.softplus(x_stats["mean"]/x_stats["max"] - y_stats["mean"]/y_stats["max"]).pow(2)
        std_loss = torch.log1m(x_stats["std"].exp() * y_stats["std"].exp()).pow(2)
        var_loss = torch.abs(x_stats["var"] - y_stats["var"]).log()
        loss = torch.sqrt(mean_loss - var_loss / std_loss)
        return loss.mean()