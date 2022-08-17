from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register
from ImageFunctions.layers import create as create_layer
from ImageFunctions.layers.activations import create as create_act
from ImageFunctions.layers import SpectralConv2d

__all__ = ['ResNeXtER', 'resnexter18', 'resnexter34', 'resnexter50', 'resnexter101', 'resnexter152']

def dwconv7x7(filters):
    return nn.Conv2d(filters, filters, kernel_size=7,
                     padding=3, groups=filters, bias=False)

def conv1x1(in_filters, out_filters):
    return nn.Linear(in_filters, out_filters)

class Block(nn.Module):
    expansion = 1

    def __init__(self, filters, drop_path=0, norm_layer=None, scale_init=1e-5):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d
            
        self.dwconv = dwconv7x7(filters)
        self.norm = norm_layer(filters)
        self.act = nn.GELU()
        self.pwconv1 = conv1x1(filters, filters * 4)
        self.pwconv2 = conv1x1(filters * 4, filters)
        self.scale = create_layer("scale", init_value=scale_init)
        self.balance = create_layer("balance")
        self.drop = create_layer("drop_path", drop_prob=drop_path)
        self.attn = create_layer("pixel_attn", f_in=filters, dropout=0.1)

    def forward(self, x):
        out = self.dwconv(x)
        out = out.permute(0,2,3,1)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = self.scale(out)
        out = out.permute(0,3,1,2)
        out = self.balance(out, self.attn(out))
        out = x + self.drop(out)

        return out

@register("resnexter")
class ResNeXtER(nn.Module):

    def __init__(self, num_colors=3, layers=[2,2,2], zero_init_residual=False, 
                 num_filters=128, out_filters=64, norm_layer=None):
        super().__init__()

        self.out_dim = out_filters

        if norm_layer is None:
            norm_layer = partial(nn.InstanceNorm2d, affine=True)
        else:
            norm_layer = eval(norm_layer) if isinstance(norm_layer, str) else norm_layer
        self._norm_layer = norm_layer

        self.filters = num_filters
        
        self.head = nn.Sequential(
            nn.Conv2d(num_colors, self.filters, 3),
            self._norm_layer(self.filters),
            nn.GELU(),
            create_layer("balanced_attn", in_planes=self.filters)
        
        )
        self.norm = norm_layer(self.filters)
        self.act = nn.GELU()
        self.attn = create_layer("balanced_attn", in_planes=self.filters)
        
        self.layer1 = self._make_layer(Block, self.filters, layers[0])
        self.layer2 = self._make_layer(Block, self.filters * 2, layers[1])
        self.layer3 = self._make_layer(Block, self.filters * 4, layers[2])
        
        self.fuse1 = nn.Conv2d(self.filters * 7, self.filters, kernel_size=7, padding=3)
        self.fuse2 = nn.Conv2d(self.filters * 7, self.filters, kernel_size=5, padding=2)
        self.fuse3 = nn.Conv2d(self.filters * 7, self.filters, kernel_size=3, padding=1)
        
        self.balance1 = create_layer("balance")
        self.balance2 = create_layer("balance")
        self.balance3 = create_layer("balance")
        
        self.gff = nn.Conv2d(self.filters, out_filters, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Block):
                    nn.init.constant_(m.norm.weight, 0)

    def _make_layer(self, block, filters, blocks, 
                    drop_path=0.1, dropout=0.2):
        """
        :param last_relu: in metric learning paradigm, the final relu is removed (last_relu = False)
        """
        norm_layer =  nn.InstanceNorm2d

        layers = list()
        layers.append(block(filters, drop_path=drop_path, norm_layer=norm_layer))
        layers.append(SpectralConv2d(filters, filters, 12, 12))
        for i in range(1, blocks):
            layers.append(block(filters, drop_path=drop_path, norm_layer=norm_layer))
            layers.append(SpectralConv2d(filters, filters, 12, 12))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)

        c1 = self.layer1(x)
        c2 = self.layer2(torch.cat([x,c1],dim=1))
        c3 = self.layer3(torch.cat([x,c1,c2],dim=1))
        
        g = torch.cat([c1, c2, c3], dim=1)
        
        f1 = self.fuse1(g)
        f2 = self.fuse2(g)
        f3 = self.fuse3(g)
        
        f = self.balance1(self.balance2(x, f1), self.balance3(f2, f3))
        
        out = self.gff(f)

        return out
