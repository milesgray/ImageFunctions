import torch
import torch.nn as nn
import torch.nn.functional as F
from .learnable import Scale, Balance
from .softmax import SpatialSoftmax2d
from .spectral import SpectralConv2d
from ImageFunctions.utility.torch import get_valid_padding
from .registry import register

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.attn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size, 
                      padding=padding, 
                      bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.attn(max_out)

class CatPoolSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, 
                      padding=padding, 
                      bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.attn(x)
    
@register("soft_wave_spatial_attn")
class SoftWaveSpatialAttention(nn.Module):
    def __init__(self, f_in: int, 
                 f_out: int=None,
                 kernel: int=3,
                 modes: int=12,
                 learn_weight: bool=True,
                 dropout: float=0.0):
        super().__init__()
        if f_out is None:
            f_out = f_in
        self.channels = f_out
        self.resize = nn.Identity()
        if f_in != f_out:
            self.resize = nn.Sequential(*[self.resize, nn.Conv2d(f_in, f_out, 1, bias=False)])

        self.spatial_conv = nn.Conv2d(f_out, 
                                      f_out, 
                                      spatial_k, 
                                      padding=get_valid_padding(spatial_k, 0), 
                                      bias=False)
        self.wave_conv = SpectralConv2d(1, f_out, 1, modes, modes)
        self.spatial_softmax = SpatialSoftmax2d(temp=1.0, requires_grad=learn_weight)
        self.use_dropout = dropout > 0
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout)
        if self.learn_weight:
            self.global_scale = Scale(1.0)
            self.wave_scale = Scale(0.0)
            self.pool_scale = Scale(0.0)
            self.global_balance = Balance()
            
        def forward(self, x):
            x = self.resize(x)
            
            spatial_y = self.spatial_conv(x)        
            spatial_attn = self.sigmoid(spatial_y)
            if self.learn_weight:
                spatial_attn = self.spatial_scale(spatial_attn)
            spatial_attn = self.spatial_softmax(spatial_attn)
            wave_attn = self.wave_conv(x)
            if self.learn_weight:
                wave_attn = self.wave_scale(wave_attn)
            wave_attn = self.sigmoid(wave_attn)            
            if self.use_dropout:
                spatial_attn = self.dropout(spatial_attn)
                wave_attn = self.dropout(wave_attn)
            attn = self.global_balance(spatial_attn, wave_attn)
            attn = self.global_scale(attn)
            return attn