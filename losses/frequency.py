import numpy as np
from numpy.fft import *
import torch
import torch.fft
import torch.fft as tfft
import torch.nn as nn
import torch.nn.functional as F

from ImageFunctions.models.layers.activations import make as make_activation
from ImageFunctions.models.layers import make as make_layer
from ImageFunctions.models import make as make_model
from .registry import register

class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, stride=1, padding=4):
        super().__init__()
        # initialize guassian kernel
        mean = (kernel_size - 1) / 2.0
        variance = (kernel_size / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size, stride=stride, padding=padding, groups=3, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)

class FilterLow(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, padding=True, include_pad=True, gaussian=False):
        super().__init__()
        if padding:
            pad = int((kernel_size - 1) / 2)
        else:
            pad = 0
        if gaussian:
            self.filter = GaussianFilter(kernel_size=kernel_size, stride=stride, padding=pad)
        else:
            self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad, count_include_pad=include_pad)
        self.recursions = recursions

    def forward(self, x):
        for i in range(self.recursions):
            x = self.filter(x)
        return x

class FilterHigh(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, include_pad=True, normalize=True, gaussian=False):
        super().__init__()
        self.filter_low = FilterLow(recursions=1, kernel_size=kernel_size, stride=stride, include_pad=include_pad,
                                    gaussian=gaussian)
        self.recursions = recursions
        self.normalize = normalize
    
    def forward(self, x):
        x = self.filter_low(x)
        x = x - self.filter_low(x)
        if self.normalize:
            return 0.5 + x * 0.5
        else:
            return x

@register("fs")
class FSLoss(nn.Module):
    def __init__(self, recursions=1, stride=1, kernel_size=5, gaussian=False):
        super().__init__()
        self.filter = FilterHigh(recursions=recursions, stride=stride, kernel_size=kernel_size, include_pad=False,
                                     gaussian=gaussian)
    def forward(self, x, y):
        x = self.filter(x)
        y = self.filter(y)
        loss = F.l1_loss(x, y)
        return loss