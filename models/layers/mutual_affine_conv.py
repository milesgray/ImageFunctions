import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from collections import OrderedDict

class MAConv(nn.Module):
    ''' Mutual Affine Convolution (MAConv) layer '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, split=2, reduction=2):
        super().__init__()
        assert split >= 2, 'Num of splits should be larger than one'

        self.num_split = split
        splits = [1 / split] * split
        self.in_split, self.in_split_rest, self.out_split = [], [], []

        for i in range(self.num_split):
            in_split = round(in_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.in_split)
            in_split_rest = in_channels - in_split
            out_split = round(out_channels * splits[i]) if i < self.num_split - 1 else in_channels - sum(self.out_split)

            self.in_split.append(in_split)
            self.in_split_rest.append(in_split_rest)
            self.out_split.append(out_split)

            setattr(self, 'fc{}'.format(i), nn.Sequential(*[
                nn.Conv2d(in_channels=in_split_rest, out_channels=int(in_split_rest // reduction), 
                          kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(in_split_rest // reduction), out_channels=in_split * 2, 
                          kernel_size=1, stride=1, padding=0, bias=True),
            ]))
            setattr(self, 'conv{}'.format(i), nn.Conv2d(in_channels=in_split, out_channels=out_split, 
                                                        kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))

    def forward(self, x):
        x = torch.split(x, self.in_split, dim=1)
        output = []

        for i in range(self.num_split):
            scale, translation = torch.split(getattr(self, 'fc{}'.format(i))(torch.cat(x[:i] + x[i + 1:], 1)),
                                             (self.in_split[i], self.in_split[i]), dim=1)
            output.append(getattr(self, 'conv{}'.format(i))(x[i] * torch.sigmoid(scale) + translation))

        return torch.cat(output, 1)
    
    
class MABlock(nn.Module):
    ''' Residual block based on MAConv '''
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True,
                 split=2, reduction=2):
        super().__init__()

        self.res = nn.Sequential(*[
            MAConv(in_channels, in_channels, kernel_size, stride, padding, bias, split, reduction),
            nn.ReLU(inplace=True),
            MAConv(in_channels, out_channels, kernel_size, stride, padding, bias, split, reduction),
        ])

    def forward(self, x):
        return x + self.res(x)
