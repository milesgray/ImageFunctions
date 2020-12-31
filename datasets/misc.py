import os
import time
import shutil
import math
from typing import Callable, Optional, Tuple, Dict, List
from argparse import Namespace

import torch
from torch import Tensor
import numpy as np

def generate_coords(batch_size: int, img_size: int) -> Tensor:
    row = torch.arange(0, img_size).float() / img_size # [img_size]
    x_coords = row.view(1, -1).repeat(img_size, 1) # [img_size, img_size]
    y_coords = x_coords.t() # [img_size, img_size]

    coords = torch.stack([x_coords, y_coords], dim=2) # [img_size, img_size, 2]
    coords = coords.view(-1, 2) # [img_size ** 2, 2]
    coords = coords.t().view(1, 2, img_size ** 2).repeat(batch_size, 1, 1) # [batch_size, 2, n_coords]

    return coords

def generate_var_sized_coords(aspect_ratios: List[float], img_size: int) -> Tensor:
    """
    Generates variable-sized coordinates for images with padding.
    This is actually done by generating "normal" coordinates, but using
    a range beyond [0, 1] for a shorter side.
    Aspect ratio is assumed to be equal to w/h.
    The goal of this functino is two constrain the spacing
    """
    coords = generate_coords(len(aspect_ratios), img_size) # [batch_size, 2, img_size ** 2]
    scales = [([1.0 / ar, 1.0] if ar < 1.0 else [1.0, ar]) for ar in aspect_ratios] # [batch_size, 2]
    scales = torch.tensor(scales).unsqueeze(2) # [batch_size, 2, 1]
    coords_scaled = coords * scales

    return coords_scaled

def generate_random_resolution_coords(batch_size: int, img_size: int, scale: float=None, min_scale: float=None) -> Tensor:
    """
    Generating random input coordinate patches.
    It's used when we train on random image patches of different resolution
    """
    assert (int(scale is None) + int(min_scale is None)) == 1, "Either scale or min_scale should be specified."

    if scale is None:
        sizes = np.random.rand(batch_size) * (1 - min_scale) + min_scale # Changing the range: [0, 1] => [min_scale, 1]
        scale = min_scale
    else:
        sizes = np.ones(batch_size) * scale # [batch_size]

    x_offsets = np.random.rand(batch_size) * (1 - scale) # [batch_size]
    y_offsets = np.random.rand(batch_size) * (1 - scale) # [batch_size]

    # Unfortunately, torch.linspace cannot operate on a batch of inputs
    x_coords = torch.from_numpy(np.linspace(x_offsets, x_offsets + sizes, img_size, dtype=np.float32)) # [img_size, batch_size]
    y_coords = torch.from_numpy(np.linspace(y_offsets, y_offsets + sizes, img_size, dtype=np.float32)) # [img_size, batch_size]

    x_coords = x_coords.view(1, img_size, batch_size).repeat(1, img_size, 1) # [img_size, img_size, batch_size]
    y_coords = y_coords.view(img_size, 1, batch_size).repeat(1, img_size, 1) # [img_size, img_size, batch_size]

    x_coords = x_coords.view(img_size ** 2, batch_size) # [img_size ** 2, batch_size]
    y_coords = y_coords.view(img_size ** 2, batch_size) # [img_size ** 2, batch_size]

    coords = torch.stack([x_coords, y_coords], dim=0).permute(2, 0, 1) # [batch_size, 2, img_size ** 2]

    return coords

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n)
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def to_pixel_samples(img, bbox=None):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
        bbox: List of tuples, coordinate ranges of img crop, [(Hv0, Hv1), (Wv0, Wv1)]
    """
    coord = make_coord(img.shape[-2:], ranges=bbox)
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb

def to_frequency_samples(f_img):
    freq = f_img.view(4, -1).permute(1, 0)
    return freq


def compute_square_padding(height: int, width: int) -> Tuple[int, int, int, int]:
    pad_l, pad_t, pad_r, pad_b = 0, 0, 0, 0

    if width < height:
        diff = height - width

        if diff % 2 == 0:
            pad_l = pad_r = diff // 2
        else:
            pad_l = 1 + diff // 2 # Left pad is 1 pixel bigger
            pad_r = diff // 2
    elif width > height:
        diff = width - height

        if diff % 2 == 0:
            pad_t = pad_b = diff // 2
        else:
            pad_t = 1 + diff // 2 # Top pad is 1 pixel bigger
            pad_b = diff // 2


    return (pad_l, pad_t, pad_r, pad_b)
