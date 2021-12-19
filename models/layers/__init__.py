from .learnable import *
from .softmax import *
from .gate import *
from .statistics import *
from .pool import *

from .activations import *

from .hessian_attn import HessianAttention, MSHF, DiEnDec, DAC
from .pixel_attn import PixelAttention
from .channel_attn import ChannelAttention, MixPoolChannelAttention
from .lhc_attn import LocalMultiHeadChannelAttention
from .triplet_attn import TripletAttention
from .balanced_attn import BalancedAttention
from .nonlocal_attn import NonLocalAttention
from .spatial_attn import SpatialAttention, CatPoolSpatialAttention
from .sd

from .cutout import MakeCutouts, WarpRandomPerspective

import functools
import torch.nn as nn
import torch.nn.functional as F

def create_activation(activation_type: str, *args, **kwargs) -> nn.Module:
    if activation_type == 'leaky_relu':
        return nn.LeakyReLU(*args, **kwargs)
    elif activation_type == 'scaled_leaky_relu':
        return ScaledLeakyReLU(*args, **kwargs)
    elif activation_type == 'relu':
        return nn.ReLU(*args, **kwargs)
    elif activation_type == 'tanh':
        return nn.Tanh(*args, **kwargs)
    elif activation_type == 'none' or activation_type is None:
        return nn.Identity(*args, **kwargs)
    elif activation_type == 'centered_sigmoid':
        return CenteredSigmoid(*args, **kwargs)
    elif activation_type == 'sine':
        return Sine(*args, **kwargs)
    elif activation_type == 'unit_centered_sigmoid':
        return UnitCenteredSigmoid(*args, **kwargs)
    elif activation_type == 'unit_centered_tanh':
        return UnitCenteredTanh(*args, **kwargs)
    elif activation_type == 'sines_cosines':
        return SinesCosines(*args, **kwargs)
    elif activation_type == 'normalizer':
        return Normalizer(*args, **kwargs)
    elif activation_type == 'normal_clip':
        return NormalClip(*args, **kwargs)
    else:
        raise NotImplementedError(f'Unknown activation type: {activation_type}')
