import os, sys, json, math, glob
import subprocess
import pathlib
from typing import cast, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
torch.backends.cudnn.benchmark = False

import kornia
import kornia.augmentation as K


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1., aspect_width=1):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cutn_zoom = int(2*cutn/3)
        self.cut_pow = cut_pow
        self.aspect_width = aspect_width
        self.transforms = None

        augmentations = []
        augmentations.append(K.RandomCrop(size=(self.cut_size,self.cut_size), p=1.0, cropping_mode="resample", return_transform=True))
        augmentations.append(WarpRandomPerspective(distortion_scale=0.40, p=0.7, return_transform=True))
        augmentations.append(K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,0.75), ratio=(0.85,1.2), cropping_mode='resample', p=0.7, return_transform=True))
        augmentations.append(K.ColorJitter(hue=0.1, saturation=0.1, p=0.8, return_transform=True))
        self.augs_zoom = nn.Sequential(*augmentations)

        augmentations = []
        if self.aspect_width == 1:
            n_s = 0.95
            n_t = (1-n_s)/2
            augmentations.append(K.RandomAffine(degrees=0, translate=(n_t, n_t), scale=(n_s, n_s), p=1.0, return_transform=True))
        elif self.aspect_width > 1:
            n_s = 1/self.aspect_width
            n_t = (1-n_s)/2
            augmentations.append(K.RandomAffine(degrees=0, translate=(0, n_t), scale=(0.9*n_s, n_s), p=1.0, return_transform=True))
        else:
            n_s = self.aspect_width
            n_t = (1-n_s)/2
            augmentations.append(K.RandomAffine(degrees=0, translate=(n_t, 0), scale=(0.9*n_s, n_s), p=1.0, return_transform=True))

        augmentations.append(K.CenterCrop(size=self.cut_size, cropping_mode='resample', p=1.0, return_transform=True))
        augmentations.append(K.RandomPerspective(distortion_scale=0.20, p=0.7, return_transform=True))
        augmentations.append(K.ColorJitter(hue=0.1, saturation=0.1, p=0.8, return_transform=True))
        self.augs_wide = nn.Sequential(*augmentations)

        self.noise_fac = 0.1

        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, x, spot=None):
        sideY, sideX = x.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        mask_indexes = None

        if spot is not None:
            spot_indexes = fetch_spot_indexes(self.cut_size, self.cut_size)
            if spot == 0:
                mask_indexes = spot_indexes[1]
            else:
                mask_indexes = spot_indexes[0]

        for _ in range(self.cutn):
            cutout = (self.av_pool(x) + self.max_pool(x))/2

            if mask_indexes is not None:
                cutout[0][mask_indexes] = 0.5

            if self.aspect_width != 1:
                if global_aspect_width > 1:
                    cutout = kornia.geometry.transform.rescale(cutout, (1, self.aspect_width))
                else:
                    cutout = kornia.geometry.transform.rescale(cutout, (1/self.aspect_width, 1))

            cutouts.append(cutout)

        if self.transforms is not None:
            batch1 = kornia.geometry.transform.warp_perspective(torch.cat(cutouts[:self.cutn_zoom], dim=0), self.transforms[:self.cutn_zoom],
                (self.cut_size, self.cut_size), padding_mode=global_padding_mode)
            batch2 = kornia.geometry.transform.warp_perspective(torch.cat(cutouts[self.cutn_zoom:], dim=0), self.transforms[self.cutn_zoom:],
                (self.cut_size, self.cut_size), padding_mode='zeros')
            batch = torch.cat([batch1, batch2])
        else:
            batch1, transforms1 = self.augs_zoom(torch.cat(cutouts[:self.cutn_zoom], dim=0))
            batch2, transforms2 = self.augs_wide(torch.cat(cutouts[self.cutn_zoom:], dim=0))

            batch = torch.cat([batch1, batch2])
            self.transforms = torch.cat([transforms1, transforms2])

        if self.verbose: print(batch.shape, self.transforms.shape)

        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch

class WarpRandomPerspective(K.RandomPerspective):
    def apply_transform(
        self, x: torch.Tensor,
        params: Dict[str, torch.Tensor],
        padding_mode: Optional[str] = "reflection",
        transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, _, height, width = x.shape
        transform = cast(torch.Tensor, transform)
        return kornia.geometry.warp_perspective(
             x, transform, (height, width),
             mode=self.resample.name.lower(),
             align_corners=self.align_corners,
             padding_mode=padding_mode)


cached_spot_indexes = {}
def fetch_spot_indexes(sideX, sideY, global_spot_file=None):
    cache_key = (sideX, sideY)

    if cache_key not in cached_spot_indexes:
        if global_spot_file is not None:
            mask_image = Image.open(global_spot_file)
        elif global_aspect_width != 1:
            mask_image = Image.open("inputs/spot_wide.png")
        else:
            mask_image = Image.open("inputs/spot_square.png")
        mask_image = mask_image.convert('RGB')
        mask_image = mask_image.resize((sideX, sideY), Image.LANCZOS)
        mask_image_tensor = TF.to_tensor(mask_image)
        mask_indexes = mask_image_tensor.ge(0.5).to(device)

        mask_indexes_off = mask_image_tensor.lt(0.5).to(device)
        cached_spot_indexes[cache_key] = [mask_indexes, mask_indexes_off]

    return cached_spot_indexes[cache_key]

def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)