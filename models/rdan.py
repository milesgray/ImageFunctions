# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

from argparse import Namespace
from functools import partial

import torch
import torch.nn as nn
from .layers import PixelAttention, NonLocalAttention
from .layers import Balance, stdv_channels

from models import register

class RDAB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, 
                 kSize=3,
                 attn_fn=PixelAttention):
        super().__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])
        self.attn = attn_fn(G)

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, self.attn(out)), 1)

class RDAB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, 
                 kSize=3,
                 attn_fn=PixelAttention):
        super().__init__()
        G0 = growRate0
        self.G = G = growRate
        C  = nConvLayers

        self.convs = []
        self.balancers = []
        self.attns = []
        for c in range(C):
            self.convs.append(RDAB_Conv(G0 + c*G, G,
                                  kSize=kSize,
                                  attn_fn=attn_fn))
            self.balancers.append(Balance(0))
            self.attns.append(PixelAttention(G0, f_out=G0 + ((c+1)*G)))
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)
        self.res_balance = Balance()

    def forward(self, x):
        output = x.clone()
        outputs = []
        leftovers = []
        for conv, balance, attn in zip(self.convs, self.balancers, self.attns):
            output = conv(output).to(x.device)
            split = torch.split(balance(output, attn(x)), self.G)
            outputs.append(split[-1])   # size: [B, G, W, H]
            leftovers.append(split[:-1])    # size: [B, G0 + (c-1)*G], W, H]
        return self.res_balance(self.LFF(torch.cat(outputs, dim=1)), x)

class RDAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        r = args.scale[0]
        nColors = args.n_colors # number of color channels expected as input
        G0 = args.G0    # baseline channel amount, also the output channel amount
        kSize = args.RDANkSize    # kernel size for SFE and GFF conv layers
        RDABkSize = args.RDABkSize    # kernel size of RDAB conv layers
        D = args.D  # number of RDAB blocks
        C = args.C  # conv layers per RDAB block
        G = args.G  # output channels of each conv layer within RDAB block, which are all concat together
        
        # configure attention layer function from string input
        attn_fn = args.attn_fn  if hasattr(args, "attn_fn") else PixelAttention        
        attn_fn = eval(attn_fn) if isinstance(attn_fn, str) else attn_fn
        
        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(nColors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        
        self.SFE_attn = attn_fn(G0)
        self.SFE_res_attn = attn_fn(G0)
        self.SFE_balance = Balance()

        # Redidual dense blocks and dense feature fusion
        self.RDABs = nn.ModuleList()
        for i in range(D):
            self.RDABs.append(
                RDAB(growRate0=G0, 
                    growRate=G, 
                    nConvLayers=C,
                    kSize=RDABkSize,
                    attn_fn=attn_fn)
            )

        # Global Feature Fusion 
        #   - This takes all of the RDAB block outputs concatenated together and
        #     combines all of the channels down to `G0`, which is the expected output
        #     channel dimension size
        self.GFF = nn.Sequential(*[
            nn.Conv2d(D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])
        
        # learns a weighted sum of GFF and SFE_res_attn
        # - final output if not upsampled
        self.GFF_balance = Balance()

        if args.no_upsampling:
            self.out_dim = G0
        else:
            self.out_dim = args.n_colors
            # Up-sampling net
            if r == 2 or r == 3:
                self.UpNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(r),
                    nn.Conv2d(G, nColors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            elif r == 4:
                self.UpNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, nColors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFE_balance(self.SFENet2(f__1), self.SFE_attn(f__1))

        RDABs_out = []
        for rdb in self.RDABs:
            x = rdb(x)
            RDABs_out.append(x)

        x = self.GFF(torch.cat(RDABs_out, dim=1))
        x = self.GFF_balance(x, self.SFE_res_attn(f__1))

        if self.args.no_upsampling:
            return x
        else:
            return self.UpNet(x)


@register('rdan')
def make_rdan(D=20, C=6, G=32, attn_fn='PixelAttention',
             G0=64, RDANkSize=3, RDABkSize=3, RDANconfig=None,
             scale=2, no_upsampling=True):
    args = Namespace()
    args.D = D
    args.C = C
    args.G = G
    args.attn_fn = attn_fn
    args.G0 = G0
    args.RDANkSize = RDANkSize
    args.RDABkSize = RDABkSize
    # original RDAN suggested parameters
    RDANstaticConfig = {
        'A': (20, 6, 32),
        'B': (16, 8, 64),
    }
    if RDANconfig in RDANstaticConfig:
        args.D, args.C, args.G = RDANstaticConfig[RDANconfig]

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.n_colors = 3
    return RDAN(args)
