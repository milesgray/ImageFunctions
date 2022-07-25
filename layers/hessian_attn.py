import torch
from torch import nn

from .statistics import get_mean_std_rgb
from .registry import register

@register("mshf")
class MSHF(nn.Module):
    def __init__(self, n_channels, kernel=3):
        super().__init__()

        pad = int((kernel - 1) / 2)

        self.grad_xx = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)
        self.grad_yy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)
        self.grad_xy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)

        for m in self.modules():
            if m == self.grad_xx:
                m.weight.data.zero_()
                m.weight.data[:, :, 1, 0] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, 1, -1] = 1
            elif m == self.grad_yy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 1] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, -1, 1] = 1
            elif m == self.grad_xy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 0] = 1
                m.weight.data[:, :, 0, -1] = -1
                m.weight.data[:, :, -1, 0] = -1
                m.weight.data[:, :, -1, -1] = 1

    def forward(self, x):
        fxx = self.grad_xx(x)
        fyy = self.grad_yy(x)
        fxy = self.grad_xy(x)
        hessian = ((fxx + fyy) + ((fxx - fyy) ** 2 + 4 * (fxy ** 2)) ** 0.5) / 2
        return hessian


@register("di_enc_dec")
class DiEnDec(nn.Module):
    def __init__(self, n_channels, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, n_channels * 2, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
            nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.Conv2d(n_channels * 4, n_channels * 8, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_channels * 8, n_channels * 4, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.ConvTranspose2d(n_channels * 2, n_channels, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
        )
        self.gate = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        output = self.gate(self.decoder(self.encoder(x)))
        return output

@register("dac")
class DAC(nn.Module):
    def __init__(self, n_channels, act=nn.ReLU(inplace=True)):
        super().__init__()

        self.mean = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, 1, 0, 1, 1, False),
            act,
            nn.Conv2d(n_channels // 16, n_channels, 1, 1, 0, 1, 1, False),
        )
        self.std = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, 1, 0, 1, 1, False),
            act,
            nn.Conv2d(n_channels // 16, n_channels, 1, 1, 0, 1, 1, False),
        )

    def forward(self, observed_feat, referred_feat):
        assert (observed_feat.size()[:2] == referred_feat.size()[:2])
        size = observed_feat.size()
        referred_mean, referred_std = get_mean_std_rgb(referred_feat)
        observed_mean, observed_std = get_mean_std_rgb(observed_feat)

        normalized_feat = (observed_feat - observed_mean.expand(
            size)) / observed_std.expand(size)
        referred_mean = self.mean(referred_mean)
        referred_std = self.std(referred_std)
        output = normalized_feat * referred_std.expand(size) + referred_mean.expand(size)
        return output

@register("hessian_attn")
class HessianAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.coder = nn.Sequential(DiEnDec(3, nn.ReLU(True)))
        self.dac = nn.Sequential(DAC(channels))
        self.hessian3 = nn.Sequential(MSHF(channels, kernel=3))
        self.hessian5 = nn.Sequential(MSHF(channels, kernel=5))
        self.hessian7 = nn.Sequential(MSHF(channels, kernel=7))

    def forward(self, x):
        hessian3 = self.hessian3(x)
        hessian5 = self.hessian5(x)
        hessian7 = self.hessian7(x)
        hessian = torch.cat((torch.mean(hessian3, dim=1, keepdim=True),
                             torch.mean(hessian5, dim=1, keepdim=True),
                             torch.mean(hessian7, dim=1, keepdim=True))
                            , 1)
        hessian = self.coder(hessian)
        attention = torch.sigmoid(self.dac[0](hessian.expand(x.size()), x))
        x = x * attention
        return x