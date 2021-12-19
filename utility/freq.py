import numpy as np
from numpy.fft import *
import torch
import torch.fft as tfft
import torch.nn as nn
import torch.nn.functional as F

def bandpass_filter(im: torch.Tensor, band_center=0.3, band_width_lower=0.1, band_width_upper=0.1):
    '''Bandpass filter the image (assumes the image is square)
    Returns
    -------
    im_bandpass: torch.Tensor
        B, C, H, W
    '''
    freq_arr = tfft.fftshift(tfft.fftfreq(n=im.shape[-1]))
    freq_arr /= np.max(np.abs(freq_arr))

    im_c = torch.stack((im, torch.zeros_like(im)),dim=4) 
    im_f = tfft.fftshift(tfft.fft(im_c, 2))
    mask_bandpass = torch.zeros(im_f.shape)

    for r in range(im_f.shape[2]):
        for c in range(im_f.shape[3]):
            dist = np.sqrt(freq_arr[r]**2 + freq_arr[c]**2)
            if dist >= band_center - band_width_lower and dist < band_center + band_width_upper:
                mask_bandpass[:, :, r, c, :] = 1
    if im.is_cuda:
        mask_bandpass = mask_bandpass.to("cuda")
    im_f_masked = torch.mul(im_f, mask_bandpass)
    im_bandpass = tfft.ifft(tfft.ifftshift(im_f_masked), 2)[...,0]

    return im_bandpass


def transform_bandpass(im: torch.Tensor, band_center=0.3, band_width_lower=0.1, band_width_upper=0.1):
    return im - bandpass_filter(im, 
                                band_center, 
                                band_width_lower, 
                                band_width_upper)


def tensor_t_augment(im: torch.Tensor, t):
    '''
    Returns
    -------
    im: torch.Tensor
        2*B, C, H, W
    ''' 
    im_copy = deepcopy(im)
    im_p = t(im)
    return torch.cat((im_copy,im_p), dim=0)  


def wavelet_filter(im: torch.Tensor, t, transform_i, idx=2, p=0.5):
    '''Filter center of highpass wavelet coeffs  
    Params
    ------
    im  : torch.Tensor 
    idx : detail coefficients ('LH':0, 'HL':1, 'HH':2)
    p   : prop to perturb coeffs
    '''
    im_t = t(im)
    # mask = torch.bernoulli((1-p) * torch.ones(im.shape[0], 5, 5))
    # im_t[1][0][:,0,idx,6:11,6:11] = im_t[1][0][:,0,idx,6:11,6:11] * mask
    im_t[1][0][:,0,idx,6:11,6:11] = 0
    return transform_i(im_t)



'''This code from https://github.com/tomrunia/PyTorchSteerablePyramid
'''
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

##########################################################################################
####### From https://github.com/v0lta/Spectral-RNN/blob/master/custom_cells.py ######
################################################################################

def hilbert(xr):
    '''
    Implements the hilbert transform, a mapping from C to R.
    Args:
        xr: The input sequence.
    Returns:
        xc: A complex sequence of the same length.
    '''    
    n = xr.shape[0]
    # Run the fft on the columns no the rows.
    x = tfft.fft(xr.t()).t()
    h = np.zeros([n])
    if n > 0 and 2*np.fix(n/2) == n:
        # even and nonempty
        h[0:int(n/2+1)] = 1
        h[1:int(n/2)] = 2
    elif n > 0:
        # odd and nonempty
        h[0] = 1
        h[1:int((n+1)/2)] = 2
    torch_h = torch.from_numpy(h)
    if len(x.shape) == 2:
        hs = np.stack([h]*x.shape[-1], -1)
        reps = x.shape()[-1]
        hs = torch.stack([torch_h]*reps, -1)
    elif len(x.shape) == 1:
        hs = torch_h
    else:
        raise NotImplementedError
    torch_hc = torch.complex(hs, torch.zeros_like(hs))
    xc = x*torch_hc
    return tfft.ifft(xc.t()).t()