import numpy as np
from numpy.fft import *
import torch
import torch.fft as tfft
import torch.nn as nn
import torch.nn.functional as F

import freq_np

def bandpass_filter(im: torch.Tensor, band_center=0.3, band_width_lower=0.1, band_width_upper=0.1):
    '''Bandpass filter the image (assumes the image is square)
    Returns
    -------
    im_bandpass: torch.Tensor
        B, C, H, W
    '''
    freq_arr = fftshift(fftfreq(n=im.shape[-1]))
    freq_arr /= np.max(np.abs(freq_arr))

    im_c = torch.stack((im, torch.zeros_like(im)),dim=4) 
    im_f = batch_fftshift2d(torch.fft(im_c, 2))
    mask_bandpass = torch.zeros(im_f.shape)

    for r in range(im_f.shape[2]):
        for c in range(im_f.shape[3]):
            dist = np.sqrt(freq_arr[r]**2 + freq_arr[c]**2)
            if dist >= band_center - band_width_lower and dist < band_center + band_width_upper:
                mask_bandpass[:, :, r, c, :] = 1
    if im.is_cuda:
        mask_bandpass = mask_bandpass.to("cuda")
    im_f_masked = torch.mul(im_f, mask_bandpass)
    im_bandpass = torch.ifft(batch_ifftshift2d(im_f_masked), 2)[...,0]

    return im_bandpass


def transform_bandpass(im: torch.Tensor, band_center=0.3, band_width_lower=0.1, band_width_upper=0.1):
    return im - bandpass_filter(im, band_center, band_width_lower, band_width_upper)


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

#########################################################################################
# SFTT LOSS - https://github.com/rishikksh20/TFGAN/blob/main/utils/stft_loss.py #######
####################################################################################

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window"):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss

#####################################################
# Discrete Cosine Transform #################
#####################################

class Dct2d(nn.Module):
    """
    Blockwhise 2D DCT
    """
    def __init__(self, blocksize=8, interleaving=False):
        """
        Parameters:
        blocksize: int, size of the Blocks for discrete cosine transform 
        interleaving: bool, should the blocks interleave?
        """
        super().__init__() # call super constructor
        
        self.blocksize = blocksize
        self.interleaving = interleaving
        
        if interleaving:
            self.stride = self.blocksize // 2
        else:
            self.stride = self.blocksize
        
        # precompute DCT weight matrix
        A = np.zeros((blocksize,blocksize))
        for i in range(blocksize):
            c_i = 1/np.sqrt(2) if i == 0 else 1.
            for n in range(blocksize):
                A[i,n] = np.sqrt(2/blocksize) * c_i * np.cos((2*n+ 1)/(blocksize*2) * i * np.pi)
        
        # set up conv layer
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32), requires_grad=False)
        self.unfold = torch.nn.Unfold(kernel_size=blocksize, padding=0, stride=self.stride)
        return
        
    def forward(self, x):
        """
        performs 2D blockwhise DCT
        
        Parameters:
        x: tensor of dimension (N, 1, h, w)
        
        Return:
        tensor of dimension (N, k, blocksize, blocksize)
        where the 2nd dimension indexes the block. Dimensions 3 and 4 are the block DCT coefficients
        """
        
        (N, C, H, W) = x.shape
        assert (C == 1), "DCT is only implemented for a single channel"
        assert (H >= self.blocksize), "Input too small for blocksize"
        assert (W >= self.blocksize), "Input too small for blocksize"
        assert (H % self.stride == 0) and (W % self.stride == 0), "FFT is only for dimensions divisible by the blocksize"
        
        # unfold to blocks
        x = self.unfold(x)
        # now shape (N, blocksize**2, k)
        (N, _, k) = x.shape
        x = x.view(-1,self.blocksize,self.blocksize,k).permute(0,3,1,2)
        # now shape (N, #k, blocksize, blocksize)
        # perform DCT
        coeff = self.A.matmul(x).matmul(self.A.transpose(0,1))
        
        return coeff
    
    def inverse(self, coeff, output_shape):
        """
        performs 2D blockwhise iDCT
        
        Parameters:
        coeff: tensor of dimension (N, k, blocksize, blocksize)
        where the 2nd dimension indexes the block. Dimensions 3 and 4 are the block DCT coefficients
        output_shape: (h, w) dimensions of the reconstructed image
        
        Return:
        tensor of dimension (N, 1, h, w)
        """
        if self.interleaving:
            raise Exception('Inverse block DCT is not implemented for interleaving blocks!')
            
        # perform iDCT
        x = self.A.transpose(0,1).matmul(coeff).matmul(self.A)
        (N, k, _, _) = x.shape
        x = x.permute(0,2,3,1).view(-1, self.blocksize**2, k)
        x = F.fold(x, output_size=(output_shape[-2], output_shape[-1]), kernel_size=self.blocksize, padding=0, stride=self.blocksize)
        return x
