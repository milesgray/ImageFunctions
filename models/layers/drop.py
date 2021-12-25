import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1. - drop_prob
    mask = torch.Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x

@register("drop_path")
class DropPath(nn.Module):
    def __init__(self, drop_prob):
        self.drop_prob = drop_prob
        if self.drop_prob > 0.:
            self.keep_prob =  1.0 - drop_rate
            self.mask = torch.Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
            
    def forward(self, x):
        if self.drop_prob > 0.:
            self.mask = torch.Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
            x /= self.keep_prob
            x *= self.mask
        return x
            