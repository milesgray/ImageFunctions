import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import SpaceToDepth

from models import register

@register("denoiser")
class Denoiser(nn.Module):
    def __init__(self, conv, channel, n_feat, act=nn.ReLU(True), bn=False):
        super().__init__()
        self.down2 = SpaceToDepth(2)
        self.down4 = SpaceToDepth(4)

        self.top1 = BasicBlock(conv, channel * 16, n_feat, 3, bn=bn)
        self.top2 = ResBlock(conv, n_feat, 3, act=act, bn=bn)
        self.top3 = BasicBlock(conv, n_feat, n_feat, 3, bn=bn)

        self.bottom1 = BasicBlock(conv, channel * 4, n_feat, 3, bn=bn)
        self.bottom_gate = conv(n_feat // 4 + n_feat, n_feat, 1)
        self.bottom2 = ResBlock(conv, n_feat, 3, act=act, bn=bn)
        self.bottom3 = BasicBlock(conv, n_feat, n_feat, 3, bn=bn)

        self.main1 = BasicBlock(conv, channel, n_feat, 3, bn=bn)
        self.main_gate = conv(n_feat + n_feat // 4, n_feat, 1)
        self.main2 = ResBlock(conv, n_feat, 3, act=act, bn=bn)
        self.main3 = BasicBlock(conv, n_feat, n_feat, 3, bn=bn)

        self.end = conv(n_feat, channel, 3)

    def fill(self, x):
        b, c, h, w = x.size()
        pad_h = 8 - h % 8
        pad_w = 8 - w % 8
        y = F.pad(x, [0, pad_w, 0, pad_h])
        return y
    
    def forward(self, x):
        b, c, h, w = x.size()
        x = fill(x)
        top_x = self.down4(x)
        bottom_x = self.down2(x)

        top_x = self.top1(top_x)
        top_x = self.top2(top_x)
        top_x = self.top3(top_x)
        top_x = F.pixel_shuffle(top_x, 2)

        bottom_x = self.bottom1(bottom_x)
        bottom_x = torch.cat((bottom_x, top_x), 1)
        bottom_x = self.bottom_gate(bottom_x)
        bottom_x = self.bottom2(bottom_x)
        bottom_x = self.bottom3(bottom_x)
        bottom_x = F.pixel_shuffle(bottom_x, 2)

        x = self.main1(x)
        x = torch.cat((x, bottom_x), 1)
        x = self.main_gate(x)
        x = self.main2(x)
        x = self.main3(x)

        x = self.end(x)
        x = x[:, :, :h, :w]
        return x