import numpy as np
from numpy.fft import *
import torch
import torch.fft
import torch.fft as tfft
import torch.nn as nn
import torch.nn.functional as F

from ImageFunctions.layers.activations import make as make_activation
from ImageFunctions.layers import make as make_layer
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


def bandpass_filter(im: torch.Tensor,
                    band_center=0.3,
                    band_width=0.1,
                    sample_spacing=None,
                    mask=None,
                    plot=False):
    '''Bandpass filter the image (assumes the image is square)

    Returns
    -------
    im_bandpass: np.ndarray
    mask: np.ndarray
        if mask is present, use this mask to set things to 1 instead of bandpass
    '''

    # find freqs
    if sample_spacing is None: # use normalized freqs [-1, 1]
        freq_arr = tfft.fftshift(
            tfft.fftfreq(n=im.shape[0])
        )
        freq_arr /= torch.max(torch.abs(freq_arr))
    else:
        sample_spacing = 0.8 # arcmins
        freq_arr = tfft.fftshift(
            tfft.fftfreq(n=im.shape[0], d=sample_spacing)
        ) # 1 / arcmin
        # print(freq_arr[0], freq_arr[-1])

    # go to freq domain
    im_f = tfft.fftshift(
        tfft.fft2(im)
    )
    if plot:
        plt.imshow(im_f.abs())
        plt.xlabel('frequency x')
        plt.ylabel('frequency y')


    # bandpass
    if mask is not None:
        assert mask.shape == im_f.shape, 'mask shape does not match shape in freq domain'
        mask_bandpass = mask
    else:
        mask_bandpass = torch.zeros(im_f.shape)
        for r in range(torch.shape[0]):
            for c in range(torch.shape[1]):
                dist = torch.sqrt(freq_arr[r]**2 + freq_arr[c]**2)
                if dist > band_center - band_width / 2 and dist < band_center + band_width / 2:
                    mask_bandpass[r, c] = 1


    im_f_masked = torch.multiply(im_f, mask_bandpass)
    if plot:
        plt.imshow(torch.abs(im_f_masked))
        plt.xticks([0, 127.5, 255], labels=[freq_arr[0].round(2), 0, freq_arr[-1].round(2)])
        plt.yticks([0, 127.5, 255], labels=[freq_arr[0].round(2), 0, freq_arr[-1].round(2)])
        plt.show()

    im_bandpass = tfft.ifft2(
        tfft.ifftshift(im_f_masked)
    )
    return im_bandpass.real