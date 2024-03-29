from argparse import Namespace
from functools import partial

import torch
import torch.nn as nn
from ImageFunctions.layers import PixelAttention, NonLocalAttention
from ImageFunctions.layers import Balance, stdv_channels
from ImageFunctions.layers import MAConv2d
from ImageFunctions.layers import create as create_layer
from ImageFunctions.layers.activations import create as create_act

from models import register

class Layer(nn.Module):
    def __init__(self, inChannels, growRate, 
                 kernel=3,
                 act="gelu",
                 attn_fn=PixelAttention):
        super().__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            MAConv2d(Cin, G, kernel, padding=(kernel-1)//2, stride=1),
            create_act(act)
        ])
        self.attn = attn_fn(G)

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, self.attn(out)), 1)

class Block(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, 
                 kernel=3,
                 act="gelu",
                 attn_fn=PixelAttention):
        super().__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(Layer(G0 + c*G, G,
                               kernel=kernel,
                               act=act,
                               attn_fn=attn_fn))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)
        self.res_balance = Balance()

    def forward(self, x):
        return self.res_balance(self.LFF(self.convs(x)), x)

class MARDAN(nn.Module):
    """ Mutually-Affine Residual Dense Attention Network
    A network of blocks that concat the results of each attention
    layer and then fuse together the result into a standard shape,
    each of which are again concat together and finally globally
    fused to produce the output.  Mutually Affine 2D Convolutions
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        r = args.scale[0]
        nColors = args.n_colors # number of color channels expected as input
        G0 = args.G0    # baseline channel amount, also the output channel amount
        kernel = args.RDNkSize    # kernel size for SFE and GFF conv layers
        RDBkSize = args.RDBkSize    # kernel size of RDB conv layers
        act = args.act  # activation function for layers
        D = args.D  # number of RDB blocks
        C = args.C  # conv layers per RDB block
        G = args.G  # output channels of each conv layer within RDB block, which are all concat together
        
        # configure attention layer function from string input
        attn_fn = args.attn_fn  if hasattr(args, "attn_fn") else PixelAttention        
        attn_fn = eval(attn_fn) if isinstance(attn_fn, str) else attn_fn
        
        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(nColors, G0, kernel, padding=(kernel-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kernel, padding=(kernel-1)//2, stride=1)
        
        self.SFE_attn = attn_fn(G0)
        self.SFE_res_attn = attn_fn(G0)
        self.SFE_balance = Balance()

        # Redidual dense blocks and dense feature fusion
        self.blocks = nn.ModuleList()
        for i in range(D):
            self.blocks.append(
                Block(growRate0=G0, 
                    growRate=G, 
                    nConvLayers=C,
                    kernel=RDBkSize,
                    act=act,
                    attn_fn=attn_fn)
            )

        # Global Feature Fusion 
        #   - This takes all of the RDB block outputs concatenated together and
        #     combines all of the channels down to `G0`, which is the expected output
        #     channel dimension size
        self.GFF = nn.Sequential(*[
            nn.Conv2d(D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kernel, padding=(kernel-1)//2, stride=1)
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

        blocks_out = []
        for block in self.blocks:
            x = block(x)
            blocks_out.append(x)

        x = self.GFF(torch.cat(blocks_out, dim=1))
        x = self.GFF_balance(x, self.SFE_res_attn(f__1))

        if self.args.no_upsampling:
            return x
        else:
            return self.UpNet(x)


@register('mardan')
def make_mardan(D=20, C=6, G=32, attn_fn='PixelAttention', act="gelu",
             G0=64, RDNkSize=3, BlockkSize=3,
             scale=2, no_upsampling=True):
    args = Namespace()
    args.D = D
    args.C = C
    args.G = G
    args.act = act
    args.attn_fn = attn_fn
    args.G0 = G0
    args.RDNkSize = RDNkSize
    args.RDBkSize = BlockkSize

    args.scale = [scale]
    args.no_upsampling = no_upsampling

    args.n_colors = 3
    return MARDAN(args)
