import torch.nn as nn
from collections import OrderedDict
import torch

from .common import sequential, get_valid_padding
from .statistics import mean_channels, stdv_channels

from .registry import register

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                     padding=padding, 
                     bias=bias, 
                     dilation=dilation,
                     groups=groups)

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer




def conv_block(in_nc, out_nc, kernel_size, 
               stride=1, 
               dilation=1, 
               groups=1, 
               bias=True,
               pad_type='zero', 
               norm_type=None, 
               act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)




@register("imdn_shortcut")
class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super().__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


# contrast-aware channel attention module
@register("imdn_contrast_attn")
class CCALayer(nn.Module):
    def __init__(self, channel: int, reduction: int=16):
        super().__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

@register("imdn_module")
class IMDModule(nn.Module):
    def __init__(self, in_channels: int, distillation_rate: float=0.25):
        super().__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.cca = CCALayer(self.distilled_channels * 4)

    def forward(self, x: Tensor):
        out_c1 = self.act(self.c1(x))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cca(out)) + x
        return out_fused

@register("imdn_moduloe_speed")
class IMDModule_speed(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super().__init__()
        self.in_channels = in_channels
        self.distilled_channels = int(self.in_channels * distillation_rate)
        self.remaining_channels = int(self.in_channels - self.distilled_channels)
        self.c1 = conv_layer(self.in_channels, self.in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, self.in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, self.in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.distilled_channels * 4, self.in_channels, 1)

    def forward(self, x):
        out_c1 = self.act(self.c1(x))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(out) + x
        return out_fused

@register("imdn_module_large")
class IMDModule_Large(nn.Module):
    def __init__(self, in_channels, distillation_rate=1/4):
        super().__init__()
        self.distilled_channels = int(in_channels * distillation_rate)  # 6
        self.remaining_channels = int(in_channels - self.distilled_channels)  # 18
        self.c1 = conv_layer(in_channels, in_channels, 3, bias=False)  # 24 --> 24
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3, bias=False)  # 18 --> 24
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3, bias=False)  # 18 --> 24
        self.c4 = conv_layer(self.remaining_channels, self.remaining_channels, 3, bias=False)  # 15 --> 15
        self.c5 = conv_layer(self.remaining_channels-self.distilled_channels, self.remaining_channels-self.distilled_channels, 3, bias=False)  # 10 --> 10
        self.c6 = conv_layer(self.distilled_channels, self.distilled_channels, 3, bias=False)  # 5 --> 5
        self.act = activation('relu')
        self.c7 = conv_layer(self.distilled_channels * 6, in_channels, 1, bias=False)

    def forward(self, x):
        out_c1 = self.act(self.c1(x))  # 24 --> 24
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1) # 6, 18
        out_c2 = self.act(self.c2(remaining_c1))  #  18 --> 24
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)  # 6, 18
        out_c3 = self.act(self.c3(remaining_c2))  # 18 --> 24
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)  # 6, 18
        out_c4 = self.act(self.c4(remaining_c3))  # 18 --> 18
        distilled_c4, remaining_c4 = torch.split(out_c4, (self.distilled_channels, self.remaining_channels-self.distilled_channels), dim=1)  # 6, 12
        out_c5 = self.act(self.c5(remaining_c4))  # 12 --> 12
        distilled_c5, remaining_c5 = torch.split(out_c5, (self.distilled_channels, self.remaining_channels-self.distilled_channels*2), dim=1)  # 6, 6
        out_c6 = self.act(self.c6(remaining_c5))  # 6 --> 6

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4, distilled_c5, out_c6], dim=1)
        out_fused = self.c7(out) + x
        return out_fused
    
