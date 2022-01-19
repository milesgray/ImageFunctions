from argparse import Namespace

import torch
import torch.nn as nn

import models
from models import register

@register('freq_disc')
class FrequencyDiscriminator(nn.Module):
    def __init__(self, model_spec):
        super().__init__()
        self.model = models.make(model_spec)
        
    def frequency_transform(self, x):
        # both images are transformed into Fourier space by applying the fast Fourier transform (FFT)
        fourier = torch.fft.fft(x, norm='ortho')
        # where we calculate amplitude and phase of all frequency components
        amp = torch.sqrt(fourier.real.pow(2) + fourier.imag.pow(2))
        phase = torch.atan2(fourier.imag, fourier.real)
        return amp, phase
    
    def forward(self, x):
        amp, phase = self.frequency_transform(x)
        freq = torch.cat([amp.view(x.shape[0], -1), phase.view(x.shape[0], -1)], dim=1)
        return self.model(freq)