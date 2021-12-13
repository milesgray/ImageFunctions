import functools
import random
import math
from PIL import Image

import numpy as np
import torch
import torch.fft as tfft
from torch.utils.data import Dataset
from torchvision import transforms
import kornia

from datasets import register
from utility import to_pixel_samples, to_frequency_samples, resize_fn
from datasets import augments

@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):
    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):
    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

class RandCropDataset(Dataset):
    def __init__(self, dataset, num_crop=2, inp_size=None, augment=False, 
                 color_augment=False, color_augment_strength=0.8):
        self.dataset = dataset
        self.num_crop = num_crop
        self.inp_size = inp_size
        self.augment = augment
        self.color_augment = color_augment
        self.color_augment_strength = color_augment_strength

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        if self.color_augment:
            img = self.apply_color_aug(img)
        
        crops = self.make_crop(img)

        if self.augment:
            crops = self.apply_augs(crops)

        result = {
            'crops': crops,
        }

        return result

    def make_crops(self, img):
        if img.shape[0] == 3:
            w_index = 1
            h_index = 2
        else:
            w_index = 0
            h_index = 1

        results = []

        for i in range(self.num_crop):
            x0 = random.randint(0, img.shape[h_index] - self.inp_size)
            y0 = random.randint(0, img.shape[w_index] - self.inp_size)
            crop = img[:, x0: x0 + self.inp_size, y0: y0 + self.inp_size]
            results.append(crop)

        return results

    def apply_color_aug(self, img):
        s = self.color_augment_strength
        color_aug_kwarg = {
            "bright": (random.random() * 2.0) * s,
            "saturation": (random.random() * 2.0)* s,
            "hue": (random.random() - 0.5) * s,
            "gamma": (random.random() * 2.0) * s,
        }
        img = augments.apply_color_distortion(img, **color_aug_kwarg)

        return img

    def apply_augs(self, crops):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        dflip = random.random() < 0.5

        def augment(x):
            if hflip:
                x = x.flip(-2)
            if vflip:
                x = x.flip(-1)
            if dflip:
                x = x.transpose(-2, -1)
            return x

        results = [augment(c) for c in crops]

        return results
    
    def shuffle_mapping(self):
        self.dataset.shuffle_mapping()

class SRRandCropDataset(RandCropDataset):
    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, color_augment=False, color_augment_strength=0.8,
                 return_hr=False):
        super().__init__(dataset, inp_size=inp_size, augment=augment, 
                         color_augment=color_augment, 
                         color_augment_strength=color_augment_strength)
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.sample_q = sample_q
        self.return_hr = return_hr

    def __getitem__(self, idx):
        img = self.dataset[idx]
        if self.color_augment:
            img = self.apply_color_aug(img)
        
        crop_lr, crop_hr = self.make_crops(img)

        if self.augment:
            crop_lr, crop_hr = self.apply_augs(crop_lr, crop_hr)

        hr_coord, hr_rgb, cell = self.create_targets(crop_hr)

        result = {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
        if self.return_hr:
            result["hr"] = crop_hr
        return result

    def make_crops(self, img):
        if img.shape[0] == 3:
            w_index = 1
            h_index = 2
        else:
            w_index = 0
            h_index = 1
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[h_index] / s + 1e-9)
            w_lr = math.floor(img.shape[w_index] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[h_index] - w_hr)
            y0 = random.randint(0, img.shape[w_index] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        return crop_lr, crop_hr

    def apply_color_aug(self, img):
        s = self.color_augment_strength
        color_aug_kwarg = {
            "bright": (random.random() * 2.0) * s,
            "saturation": (random.random() * 2.0)* s,
            "hue": (random.random() - 0.5) * s,
            "gamma": (random.random() * 2.0) * s,
        }
        img = augments.apply_color_distortion(img, **color_aug_kwarg)

        return img

    def apply_augs(self, crop_lr, crop_hr):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        dflip = random.random() < 0.5

        def augment(x):
            if hflip:
                x = x.flip(-2)
            if vflip:
                x = x.flip(-1)
            if dflip:
                x = x.transpose(-2, -1)
            return x

        crop_lr = augment(crop_lr)
        crop_hr = augment(crop_hr)

        return crop_lr, crop_hr

    def create_targets(self, crop_hr):
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return hr_coord, hr_rgb, cell
    
    def shuffle_mapping(self):
        self.dataset.shuffle_mapping()

@register('sr-explicit-downsampled-randcrop')
class SRExplicitDownsampledRandCrop(SRRandCropDataset):
    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, color_augment=False, color_augment_strength=0.8,
                 return_hr=False):
        super().__init__(dataset, inp_size=inp_size, scale_min=scale_min, scale_max=scale_max,
                 augment=augment, sample_q=sample_q, color_augment=color_augment, color_augment_strength=color_augment_strength,
                 return_hr=return_hr)

@register('sr-randrange-downsampled-randcrop')
class SRRandRangeDownsampledRandCrop(SRRandCropDataset):
    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, vary_q=False, color_augment=False, color_augment_strength=0.8,
                 return_hr=False):
        super().__init__(dataset, inp_size=inp_size, scale_min=scale_min, scale_max=scale_max,
                 augment=augment, sample_q=sample_q, color_augment=color_augment, color_augment_strength=color_augment_strength,
                 return_hr=return_hr)
        self.vary_q = vary_q

    def make_crops(self, img):
        if img.shape[0] == 3:
            w_index = 1
            h_index = 2
        else:
            w_index = 0
            h_index = 1
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[w_index] / s + 1e-9)
            w_lr = math.floor(img.shape[h_index] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = round(random.uniform(min(min(self.inp_size*s, img.shape[w_index]), img.shape[h_index]) // s, 
                                        min(min(self.inp_size*s*s, img.shape[w_index]), img.shape[h_index]) // s))
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[w_index] - w_hr)
            y0 = random.randint(0, img.shape[h_index] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        return crop_lr, crop_hr

    def create_targets(self, crop_hr):
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            if self.vary_q:
                sample_lst = np.random.choice(len(hr_coord), 
                                              min(round(self.sample_q * s), len(hr_coord)), 
                                              replace=False)
            else:
                sample_lst = np.random.choice(len(hr_coord), 
                                              self.sample_q, 
                                              replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return hr_coord, hr_rgb, cell

@register('sr-setrange-downsampled-randcrop')
class SRSetRangeDownsampledRandCrop(SRRandCropDataset):
    def __init__(self, dataset, 
                 inp_size=None, inp_size_min=None, inp_size_max=None, min_size=16,
                 scale_min=1, scale_max=None,
                 augment=False, color_augment=False, color_augment_strength=0.8, 
                 sample_q=None, vary_q=False, max_q=None, use_subgrid_coords=False,
                 return_hr=False, resize_hr=False, return_freq=False):
        super().__init__(dataset, inp_size=inp_size, scale_min=scale_min, scale_max=scale_max,
                 augment=augment, sample_q=sample_q, color_augment=color_augment, 
                 color_augment_strength=color_augment_strength,
                 return_hr=return_hr)
        self.inp_size_min = inp_size_min
        self.inp_size_max = inp_size_max
        self.min_size = min_size
        self.resize_hr = resize_hr
        self.return_freq = return_freq
        self.vary_q = vary_q
        self.max_q = max_q
        self.use_subgrid_coords = use_subgrid_coords
        self.rand_scale = None

    def __getitem__(self, idx):
        img = self.dataset[idx]
        if self.color_augment:
            img = self.apply_color_aug(img)
        
        crop_lr, crop_hr, f_crop_hr, grid_crop_hr = self.make_crops(img)

        if self.augment:
            crop_lr, crop_hr, f_crop_hr, grid_crop_hr = self.apply_augs(crop_lr, crop_hr, f_crop_hr, grid_crop_hr)

        hr_coord, hr_rgb, cell, hr_freq = self.create_targets(crop_hr, f_crop_hr, grid_crop_hr)

        result = {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
        if self.return_freq:
            result['f_gt'] = hr_freq
        if self.return_hr:
            result["hr"] = crop_hr
        return result

    def make_crops(self, img):
        grid = kornia.utils.create_meshgrid(img.shape[1], img.shape[2]).squeeze()
        s = self.rand_scale = random.uniform(self.scale_min, self.scale_max)
        s = max(1, s)
        if img.shape[0] == 3:
            h_index = 1
            w_index = 2
        else:
            h_index = 0
            w_index = 1
        img_width = img.shape[w_index]
        img_height = img.shape[h_index]
        rand_range_min = round(min(min(round(self.inp_size_min*s), img_width), img_height) / s)
        rand_range_max = round(min(min(round(self.inp_size_max*s), img_width), img_height) / s)
        rand_range_min = max(rand_range_min, self.min_size)
        rand_range_max = max(rand_range_max, self.min_size)
        w_lr = round(random.uniform(rand_range_min, 
                                    rand_range_max))
        w_lr = max(self.min_size, w_lr)
        w_hr = max(round(self.min_size * s), round(w_lr * s))
        if img_height - w_hr < self.min_size or img_width - w_hr < self.min_size:
            w_lr = self.min_size
            w_hr = round(w_lr * s)
        x0 = random.randint(0, max(img_height - w_hr, 0))
        y0 = random.randint(0, max(img_width - w_hr, 0))
        x0 = min(img_height - w_hr, x0)
        y0 = min(img_width - w_hr, y0)
        
        if img.shape[0] == 3:
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        else:
            crop_hr = img[x0: x0 + w_hr, y0: y0 + w_hr, :]
        grid_crop_hr = grid[x0: x0 + w_hr, y0: y0 + w_hr, :]
        if self.return_freq:
            f_img = tfft.hfft(img.movedim((0,1,2),(2,0,1)), norm="ortho").movedim((0,1,2),(1,2,0))
            f_crop_hr = f_img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            if self.resize_hr:
                if self.inp_size is None:
                    f_crop_hr = resize_fn(f_crop_hr, round(w_lr * s))
                else:
                    f_crop_hr = resize_fn(f_crop_hr, round(self.inp_size * s))
        else:
            f_crop_hr = None
        if self.inp_size is None:
            try:
                if crop_hr.shape[h_index] <= w_lr or crop_hr.shape[w_index] <= w_lr:
                    print(f"0 Bad shape: {crop_hr.shape}, scale: {s}, low res size: {w_lr}, img size: {img.shape}, width: {img_width} height: {img_height}, x0: {x0}, y0: {y0}")
                crop_lr = resize_fn(crop_hr, w_lr)
            except Exception as e:
                print(f"1 Bad shape: {crop_hr.shape}, scale: {s}, low res size: {w_lr}, img size: {img.shape}, width: {img_width} height: {img_height}, x0: {x0}, y0: {y0}")
                if all([d > 0 for d in crop_hr.shape]):
                    if self.min_size > img_width or self.min_size > img_height:
                        if w_hr <= 0:
                            w_hr = self.min_size
                        if x0 <= 0:
                            x0 = 0
                        if y0 <= 0:
                            y0 = 0
                        if img.shape[0] == 3:
                            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
                        else:
                            crop_hr = img[x0: x0 + w_hr, y0: y0 + w_hr, :]
                    crop_lr = resize_fn(crop_hr, round(self.min_size * s))
                else:
                    if w_hr <= 0:
                        w_hr = self.min_size
                    if x0 <= 0:
                        x0 = 0
                    if y0 <= 0:
                        y0 = 0
                    if img.shape[0] == 3:
                        crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
                    else:
                        crop_hr = img[x0: x0 + w_hr, y0: y0 + w_hr, :]
                    crop_lr = resize_fn(crop_hr, round(self.min_size * s))
        else:
            if crop_hr.shape[h_index] < self.inp_size or crop_hr.shape[w_index] < self.inp_size:
                print(f"2 Bad shape: {crop_hr.shape}, scale: {s}, low res size: {w_lr}, img size: {img.shape}, width: {img_width} height: {img_height}, x0: {x0}, y0: {y0}")
            crop_lr = resize_fn(crop_hr, self.inp_size)
        if self.resize_hr:
            if self.inp_size is None:
                crop_hr = resize_fn(crop_hr, round(w_lr * s))
            else:
                crop_hr = resize_fn(crop_hr, round(self.inp_size * s))

        return crop_lr, crop_hr, f_crop_hr, grid_crop_hr

    def apply_augs(self, crop_lr, crop_hr, f_crop_hr, grid_crop_hr):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        dflip = random.random() < 0.5

        def augment(x):
            if hflip:
                x = x.flip(-2)
            if vflip:
                x = x.flip(-1)
            if dflip:
                x = x.transpose(-2, -1)
            return x

        crop_lr = augment(crop_lr)
        crop_hr = augment(crop_hr)
        grid_crop_hr = augment(grid_crop_hr)
        if self.return_freq:
            f_crop_hr = augment(f_crop_hr)

        return crop_lr, crop_hr, f_crop_hr, grid_crop_hr

    def create_targets(self, crop_hr, f_crop_hr, grid_crop_hr):
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        if self.return_freq:
            hr_freq = to_frequency_samples(f_crop_hr.contiguous())
        else:
            hr_freq = None

        if self.sample_q is not None:
            if self.vary_q:
                max_q = len(hr_coord) if self.max_q is None else self.max_q
                sample_q = round(self.sample_q * self.rand_scale)
                sample_lst = np.random.choice(len(hr_coord), 
                                              min(min(max_q, sample_q), len(hr_coord)),
                                              replace=False)
            else:
                sample_lst = np.random.choice(len(hr_coord), 
                                              min(self.sample_q, len(hr_coord)), 
                                              replace=False)
            if self.use_subgrid_coords:
                hr_coord = grid_crop_hr.view(-1, 2)[sample_lst]
            else:
                hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
            if self.return_freq:
                hr_freq = hr_freq[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return hr_coord, hr_rgb, cell, hr_freq

@register('zr-setrange-downsampled-randcrop')
class ZRSetRangeDownsampledRandCrop(SRSetRangeDownsampledRandCrop):
    def make_crops(self, img):
        if img.shape[0] == 3:
            w_index = 1
            h_index = 2
        else:
            w_index = 0
            h_index = 1
        s = random.uniform(self.scale_min, self.scale_max)

        w_lr = round(random.uniform(min(min(self.inp_size_min*s, img.shape[w_index]), img.shape[h_index]) // s, 
                                    min(min(self.inp_size_max*s, img.shape[w_index]), img.shape[h_index]) // s))
        w_hr = round(w_lr * s)
        x0 = random.randint(0, img.shape[w_index] - w_hr)
        y0 = random.randint(0, img.shape[h_index] - w_hr)
        crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        if self.return_freq:
            f_img = tfft.hfft(img.movedim((0,1,2),(2,0,1)), norm="ortho").movedim((0,1,2),(1,2,0))
            f_crop_hr = f_img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            if self.resize_hr:
                if self.inp_size is None:
                    f_crop_hr = resize_fn(f_crop_hr, round(w_lr * s))
                else:
                    f_crop_hr = resize_fn(f_crop_hr, round(self.inp_size * s))
        if self.inp_size is None:
            crop_lr = img[:, x0 + round(w_lr/s): x0 + w_lr + round(w_lr/s), y0 + round(w_lr/s): y0 + w_lr + round(w_lr/s)]
        else:
            crop_lr = resize_fn(img[:, x0 + round(w_lr/s): x0 + w_lr + round(w_lr/s), y0 + round(w_lr/s): y0 + w_lr + round(w_lr/s)], self.inp_size)
        if self.resize_hr:
            if self.inp_size is None:
                crop_hr = resize_fn(crop_hr, round(w_lr * s))
            else:
                crop_hr = resize_fn(crop_hr, round(self.inp_size * s))
        return crop_lr, crop_hr, f_crop_hr

@register('ed-setrange-downsampled-randcrop')
class EDSetRangeDownsampledRandCrop(SRRandCropDataset):
    def __init__(self, dataset, 
                 inp_size=None, inp_size_min=None, inp_size_max=None, crop_size=32,
                 augment=False, color_augment=False, color_augment_strength=0.8, 
                 sample_q=None, vary_q=False,
                 use_subgrid_coords=False, use_rgb_grayscale=False,
                 return_hr=False, resize_hr=False, return_freq=False):
        super().__init__(dataset, inp_size=inp_size,
                 augment=augment, sample_q=sample_q, color_augment=color_augment, 
                 color_augment_strength=color_augment_strength,
                 return_hr=return_hr)
        self.inp_size_min = inp_size_min
        self.inp_size_max = inp_size_max
        self.crop_size = crop_size
        self.resize_hr = resize_hr
        self.return_freq = return_freq
        self.vary_q = vary_q
        self.use_subgrid_coords = use_subgrid_coords
        self.use_rgb_grayscale = use_rgb_grayscale
        self.out_channels = 3 if use_rgb_grayscale else 1
        self.rand_scale = None


    def __getitem__(self, idx):
        img, edge_img = self.dataset[idx]
        if self.color_augment:
            img = self.apply_color_aug(img)
        
        crop_lr, crop_hr, f_crop_hr, grid_crop_hr = self.make_crops(img, edge_img)        

        if self.augment:
            crop_lr, crop_hr, f_crop_hr, grid_crop_hr = self.apply_augs(crop_lr, crop_hr, f_crop_hr, grid_crop_hr)

        self.rand_scale = random.uniform(0.7, 1.3)
        if self.use_rgb_grayscale: 
            if len(crop_hr.size()) == 2 or crop_hr.shape[0] == 1:
                crop_hr = torch.stack([crop_hr, crop_hr, crop_hr], dim=0)
        hr_coord, hr_rgb, cell, hr_freq = self.create_targets(crop_hr, f_crop_hr, grid_crop_hr)

        result = {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
        if self.return_freq:
            result['f_gt'] = hr_freq
        if self.return_hr:
            result["hr"] = crop_hr
        return result

    def make_crops(self, img, edge_img, set_scale=None, set_range=None, return_rand_vals=False):
        grid = kornia.utils.create_meshgrid(img.shape[1], img.shape[2]).squeeze()
        
        if img.shape[0] == 3:
            h_index = 1
            w_index = 2
        else:
            h_index = 0
            w_index = 1

        if len(edge_img.size()) == 2:
            h_index_ed = 0
            w_index_ed = 1
        else:
            if edge_img.shape[0] == 1 or edge_img.shape[0] == 3:
                h_index_ed = 1
                w_index_ed = 2
            else:
                h_index_ed = 0
                w_index_ed = 1

        img_width = img.shape[w_index]
        img_height = img.shape[h_index]
        
        w_gt = self.crop_size
        w_gt = max(self.inp_size_min, w_gt)
        w_ed = max(self.inp_size_min, w_gt)
        if img_height - w_ed < self.inp_size_min or img_width - w_ed < self.inp_size_min:
            w_gt = self.inp_size_min
            w_ed = w_gt
        x0_val = random.randint(0, max(img_height - w_ed, 0))
        y0_val = random.randint(0, max(img_width - w_ed, 0))
        x0 = min(img_height - w_ed, x0_val)
        y0 = min(img_width - w_ed, y0_val)
        
        if img.shape[0] == 3:
            crop_ed = edge_img[:, x0: x0 + w_ed, y0: y0 + w_ed]
        else:
            crop_ed = edge_img[x0: x0 + w_ed, y0: y0 + w_ed, :]
        
        grid_crop_ed = grid[x0: x0 + w_ed, y0: y0 + w_ed, :]
        
        if self.return_freq:
            f_edge_img = tfft.hfft(edge_img.movedim((0,1,2),(2,0,1)), norm="ortho").movedim((0,1,2),(1,2,0))
            f_crop_ed = f_edge_img[:, x0: x0 + w_ed, y0: y0 + w_ed]            
        else:
            f_crop_ed = None
        
        if self.inp_size is None:
            try:
                #if crop_ed.shape[h_index_ed] <= w_gt or crop_ed.shape[w_index_ed] <= w_gt:
                #    print(f"0 Bad shape: {crop_ed.shape}, low res size: {w_gt}, img size: {img.shape}, width: {img_width} height: {img_height}, x0: {x0}, y0: {y0}")
                if img.shape[0] == 3:
                    crop_gt = img[:, x0: x0 + w_gt, y0: y0 + w_gt]
                else:
                    crop_gt = img[x0: x0 + w_gt, y0: y0 + w_gt, :]
            except Exception as e:
                print(f"1 Bad shape: {crop_ed.shape}, low res size: {w_gt}, img size: {img.shape}, width: {img_width} height: {img_height}, x0: {x0}, y0: {y0}")
                if all([d > 0 for d in crop_ed.shape]):
                    if self.inp_size_min > img_width or self.inp_size_min > img_height:
                        if w_ed <= 0:
                            w_ed = self.inp_size_min
                        if x0 <= 0:
                            x0 = 0
                        if y0 <= 0:
                            y0 = 0
                        if len(edge_img.shape) == 2:
                            crop_ed = edge_img[x0: x0 + w_ed, y0: y0 + w_ed]
                        else:
                            if edge_img.shape[0] == 1:
                                crop_ed = edge_img[:, x0: x0 + w_ed, y0: y0 + w_ed]
                            else:
                                crop_ed = edge_img[x0: x0 + w_ed, y0: y0 + w_ed, :]
                    if img.shape[0] == 3:
                        crop_gt = img[:, x0: x0 + w_gt, y0: y0 + w_gt]
                    else:
                        crop_gt = img[x0: x0 + w_gt, y0: y0 + w_gt, :]
                else:
                    if w_ed <= 0:
                        w_ed = self.inp_size_min
                    if x0 <= 0:
                        x0 = 0
                    if y0 <= 0:
                        y0 = 0
                    if len(edge_img.shape) == 2:
                        crop_ed = edge_img[x0: x0 + w_ed, y0: y0 + w_ed]
                    else:
                        if edge_img.shape[0] == 1:
                            crop_ed = edge_img[:, x0: x0 + w_ed, y0: y0 + w_ed]
                        else:
                            crop_ed = edge_img[x0: x0 + w_ed, y0: y0 + w_ed, :]
                    if img.shape[0] == 3:
                        crop_gt = img[:, x0: x0 + w_gt, y0: y0 + w_gt]
                    else:
                        crop_gt = img[x0: x0 + w_gt, y0: y0 + w_gt, :]
        else:
            if crop_ed.shape[h_index] < self.inp_size or crop_ed.shape[w_index] < self.inp_size:
                print(f"2 Bad shape: {crop_ed.shape}, ground truth size: {w_gt}, edge_img size: {edge_img.shape}, width: {img_width} height: {img_height}, x0: {x0}, y0: {y0}")
            if img.shape[0] == 3:
                crop_gt = img[:, x0: x0 + w_gt, y0: y0 + w_gt]
            else:
                crop_gt = img[x0: x0 + w_gt, y0: y0 + w_gt, :]

        return crop_gt, crop_ed, f_crop_ed, grid_crop_ed

    def apply_augs(self, crop_lr, crop_hr, f_crop_hr, grid_crop_hr):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        dflip = random.random() < 0.5

        def augment(x):
            if hflip:
                x = x.flip(-2)
            if vflip:
                x = x.flip(-1)
            if dflip:
                x = x.transpose(-2, -1)
            return x

        crop_lr = augment(crop_lr)
        crop_hr = augment(crop_hr)
        grid_crop_hr = augment(grid_crop_hr)
        if self.return_freq:
            f_crop_hr = augment(f_crop_hr)

        return crop_lr, crop_hr, f_crop_hr, grid_crop_hr

    def create_targets(self, crop_hr, f_crop_hr, grid_crop_hr):
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous(), channels=self.out_channels)
        if self.return_freq:
            hr_freq = to_frequency_samples(f_crop_hr.contiguous())
        else:
            hr_freq = None

        if self.sample_q is not None:
            if self.vary_q:
                sample_lst = np.random.choice(len(hr_coord), 
                                              min(round(self.sample_q * self.rand_scale), len(hr_coord)), 
                                              replace=False)
            else:
                sample_lst = np.random.choice(len(hr_coord), 
                                              min(self.sample_q, len(hr_coord)), 
                                              replace=False)
            if self.use_subgrid_coords:
                hr_coord = grid_crop_hr.view(-1, 2)[sample_lst]
            else:
                hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
            if self.return_freq:
                hr_freq = hr_freq[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return hr_coord, hr_rgb, cell, hr_freq

@register('contrastive-randcrop')
class ContrastiveRandCrop(RandCropDataset):
    def __init__(self, dataset, 
                 inp_size=None, inp_size_min=None, inp_size_max=None, crop_size=32,
                 augment=False, color_augment=False, color_augment_strength=0.8, 
                 sample_q=None, vary_q=False,
                 use_subgrid_coords=False, use_rgb_grayscale=False,
                 return_hr=False, resize_hr=False, return_freq=False):
        super().__init__(dataset, inp_size=inp_size,
                 augment=augment, sample_q=sample_q, color_augment=color_augment, 
                 color_augment_strength=color_augment_strength,
                 return_hr=return_hr)
        self.inp_size_min = inp_size_min
        self.inp_size_max = inp_size_max
        self.crop_size = crop_size
        self.resize_hr = resize_hr
        self.return_freq = return_freq
        self.vary_q = vary_q
        self.use_subgrid_coords = use_subgrid_coords
        self.use_rgb_grayscale = use_rgb_grayscale
        self.out_channels = 3 if use_rgb_grayscale else 1
        self.rand_scale = None


    def __getitem__(self, idx):
        img, edge_img = self.dataset[idx]
        if self.color_augment:
            img = self.apply_color_aug(img)
        
        crop_lr, crop_hr, f_crop_hr, grid_crop_hr = self.make_crops(img, edge_img)        

        if self.augment:
            crop_lr, crop_hr, f_crop_hr, grid_crop_hr = self.apply_augs(crop_lr, crop_hr, f_crop_hr, grid_crop_hr)

        self.rand_scale = random.uniform(0.7, 1.3)
        if self.use_rgb_grayscale: 
            if len(crop_hr.size()) == 2 or crop_hr.shape[0] == 1:
                crop_hr = torch.stack([crop_hr, crop_hr, crop_hr], dim=0)
        hr_coord, hr_rgb, cell, hr_freq = self.create_targets(crop_hr, f_crop_hr, grid_crop_hr)

        result = {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
        if self.return_freq:
            result['f_gt'] = hr_freq
        if self.return_hr:
            result["hr"] = crop_hr
        return result

    def make_crops(self, img, edge_img, set_scale=None, set_range=None, return_rand_vals=False):
        grid = kornia.utils.create_meshgrid(img.shape[1], img.shape[2]).squeeze()
        
        if img.shape[0] == 3:
            h_index = 1
            w_index = 2
        else:
            h_index = 0
            w_index = 1

        if len(edge_img.size()) == 2:
            h_index_ed = 0
            w_index_ed = 1
        else:
            if edge_img.shape[0] == 1 or edge_img.shape[0] == 3:
                h_index_ed = 1
                w_index_ed = 2
            else:
                h_index_ed = 0
                w_index_ed = 1

        img_width = img.shape[w_index]
        img_height = img.shape[h_index]
        
        w_gt = self.crop_size
        w_gt = max(self.inp_size_min, w_gt)
        w_ed = max(self.inp_size_min, w_gt)
        if img_height - w_ed < self.inp_size_min or img_width - w_ed < self.inp_size_min:
            w_gt = self.inp_size_min
            w_ed = w_gt
        x0_val = random.randint(0, max(img_height - w_ed, 0))
        y0_val = random.randint(0, max(img_width - w_ed, 0))
        x0 = min(img_height - w_ed, x0_val)
        y0 = min(img_width - w_ed, y0_val)
        
        if img.shape[0] == 3:
            crop_ed = edge_img[:, x0: x0 + w_ed, y0: y0 + w_ed]
        else:
            crop_ed = edge_img[x0: x0 + w_ed, y0: y0 + w_ed, :]
        
        grid_crop_ed = grid[x0: x0 + w_ed, y0: y0 + w_ed, :]
        
        if self.return_freq:
            f_edge_img = tfft.hfft(edge_img.movedim((0,1,2),(2,0,1)), norm="ortho").movedim((0,1,2),(1,2,0))
            f_crop_ed = f_edge_img[:, x0: x0 + w_ed, y0: y0 + w_ed]            
        else:
            f_crop_ed = None
        
        if self.inp_size is None:
            try:
                #if crop_ed.shape[h_index_ed] <= w_gt or crop_ed.shape[w_index_ed] <= w_gt:
                #    print(f"0 Bad shape: {crop_ed.shape}, low res size: {w_gt}, img size: {img.shape}, width: {img_width} height: {img_height}, x0: {x0}, y0: {y0}")
                if img.shape[0] == 3:
                    crop_gt = img[:, x0: x0 + w_gt, y0: y0 + w_gt]
                else:
                    crop_gt = img[x0: x0 + w_gt, y0: y0 + w_gt, :]
            except Exception as e:
                print(f"1 Bad shape: {crop_ed.shape}, low res size: {w_gt}, img size: {img.shape}, width: {img_width} height: {img_height}, x0: {x0}, y0: {y0}")
                if all([d > 0 for d in crop_ed.shape]):
                    if self.inp_size_min > img_width or self.inp_size_min > img_height:
                        if w_ed <= 0:
                            w_ed = self.inp_size_min
                        if x0 <= 0:
                            x0 = 0
                        if y0 <= 0:
                            y0 = 0
                        if len(edge_img.shape) == 2:
                            crop_ed = edge_img[x0: x0 + w_ed, y0: y0 + w_ed]
                        else:
                            if edge_img.shape[0] == 1:
                                crop_ed = edge_img[:, x0: x0 + w_ed, y0: y0 + w_ed]
                            else:
                                crop_ed = edge_img[x0: x0 + w_ed, y0: y0 + w_ed, :]
                    if img.shape[0] == 3:
                        crop_gt = img[:, x0: x0 + w_gt, y0: y0 + w_gt]
                    else:
                        crop_gt = img[x0: x0 + w_gt, y0: y0 + w_gt, :]
                else:
                    if w_ed <= 0:
                        w_ed = self.inp_size_min
                    if x0 <= 0:
                        x0 = 0
                    if y0 <= 0:
                        y0 = 0
                    if len(edge_img.shape) == 2:
                        crop_ed = edge_img[x0: x0 + w_ed, y0: y0 + w_ed]
                    else:
                        if edge_img.shape[0] == 1:
                            crop_ed = edge_img[:, x0: x0 + w_ed, y0: y0 + w_ed]
                        else:
                            crop_ed = edge_img[x0: x0 + w_ed, y0: y0 + w_ed, :]
                    if img.shape[0] == 3:
                        crop_gt = img[:, x0: x0 + w_gt, y0: y0 + w_gt]
                    else:
                        crop_gt = img[x0: x0 + w_gt, y0: y0 + w_gt, :]
        else:
            if crop_ed.shape[h_index] < self.inp_size or crop_ed.shape[w_index] < self.inp_size:
                print(f"2 Bad shape: {crop_ed.shape}, ground truth size: {w_gt}, edge_img size: {edge_img.shape}, width: {img_width} height: {img_height}, x0: {x0}, y0: {y0}")
            if img.shape[0] == 3:
                crop_gt = img[:, x0: x0 + w_gt, y0: y0 + w_gt]
            else:
                crop_gt = img[x0: x0 + w_gt, y0: y0 + w_gt, :]

        return crop_gt, crop_ed, f_crop_ed, grid_crop_ed

    def apply_augs(self, crop_lr, crop_hr, f_crop_hr, grid_crop_hr):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        dflip = random.random() < 0.5

        def augment(x):
            if hflip:
                x = x.flip(-2)
            if vflip:
                x = x.flip(-1)
            if dflip:
                x = x.transpose(-2, -1)
            return x

        crop_lr = augment(crop_lr)
        crop_hr = augment(crop_hr)
        grid_crop_hr = augment(grid_crop_hr)
        if self.return_freq:
            f_crop_hr = augment(f_crop_hr)

        return crop_lr, crop_hr, f_crop_hr, grid_crop_hr

    def create_targets(self, crop_hr, f_crop_hr, grid_crop_hr):
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous(), channels=self.out_channels)
        if self.return_freq:
            hr_freq = to_frequency_samples(f_crop_hr.contiguous())
        else:
            hr_freq = None

        if self.sample_q is not None:
            if self.vary_q:
                sample_lst = np.random.choice(len(hr_coord), 
                                              min(round(self.sample_q * self.rand_scale), len(hr_coord)), 
                                              replace=False)
            else:
                sample_lst = np.random.choice(len(hr_coord), 
                                              min(self.sample_q, len(hr_coord)), 
                                              replace=False)
            if self.use_subgrid_coords:
                hr_coord = grid_crop_hr.view(-1, 2)[sample_lst]
            else:
                hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
            if self.return_freq:
                hr_freq = hr_freq[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return hr_coord, hr_rgb, cell, hr_freq

#################################################################################################################3
######### O   L   D    0000  V  E  R  S  I  O  N  S  ##################################
#####################################################

class SRSetRangeDownsampledRandCrop___old(Dataset):
    def __init__(self, dataset, inp_size=None, inp_size_min=None, inp_size_max=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, vary_q=False, 
                 color_augment=False, color_augment_strength=0.8, 
                 return_hr=False, resize_hr=False, return_freq=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.inp_size_min = inp_size_min
        self.inp_size_max = inp_size_max
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.color_augment = color_augment
        self.color_augment_strength = color_augment_strength
        self.sample_q = sample_q
        self.vary_q = vary_q
        self.return_hr = return_hr
        self.resize_hr = resize_hr
        self.return_freq = return_freq

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        grid = kornia.utils.create_meshgrid(img.shape[1], img.shape[2]).squeeze()
        
        if self.color_augment:
            s = self.color_augment_strength
            color_aug_kwarg = {
                "bright": (random.random() * 2.0) * s,
                "saturation": (random.random() * 2.0)* s,
                "hue": (random.random() - 0.5) * s,
                "gamma": (random.random() * 2.0) * s,
            }
            img = augments.apply_color_distortion(img, **color_aug_kwarg)
        s = random.uniform(self.scale_min, self.scale_max)

        w_lr = round(random.uniform(min(min(self.inp_size_min*s, img.shape[-2]), img.shape[-1]) // s, 
                                    min(min(self.inp_size_max*s, img.shape[-2]), img.shape[-1]) // s))
        w_hr = round(w_lr * s)
        x0 = random.randint(0, img.shape[-2] - w_hr)
        y0 = random.randint(0, img.shape[-1] - w_hr)
        crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        grid_crop_hr = grid[x0: x0 + w_hr, y0: y0 + w_hr, :]
        if self.return_freq:
            f_img = tfft.hfft(img.movedim((0,1,2),(2,0,1)), norm="ortho").movedim((0,1,2),(1,2,0))
            f_crop_hr = f_img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            if self.resize_hr:
                if self.inp_size is None:
                    f_crop_hr = resize_fn(f_crop_hr, round(w_lr * s))
                else:
                    f_crop_hr = resize_fn(f_crop_hr, round(self.inp_size * s))
        if self.inp_size is None:
            crop_lr = resize_fn(crop_hr, w_lr)
        else:
            crop_lr = resize_fn(crop_hr, self.inp_size)
        if self.resize_hr:
            if self.inp_size is None:
                crop_hr = resize_fn(crop_hr, round(w_lr * s))
            else:
                crop_hr = resize_fn(crop_hr, round(self.inp_size * s))

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
            if self.return_freq:
                f_crop_hr = augment(f_crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        if self.return_freq:
            hr_freq = to_frequency_samples(f_crop_hr.contiguous())

        if self.sample_q is not None:
            if self.vary_q:
                sample_lst = np.random.choice(len(hr_coord), 
                                              min(round(self.sample_q * s), len(hr_coord)), 
                                              replace=False)
            else:
                sample_lst = np.random.choice(len(hr_coord), 
                                              min(self.sample_q, len(hr_coord)), 
                                              replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
            if self.return_freq:
                hr_freq = hr_freq[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        result = {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
        }
        if self.return_freq:
            result['f_gt'] = hr_freq
        if self.return_hr:
            result["hr"] = crop_hr
        return result

    def shuffle_mapping(self):
        self.dataset.shuffle_mapping()


class ZRSetRangeDownsampledRandCrop____old(Dataset):
    def __init__(self, dataset, inp_size=None, inp_size_min=None, inp_size_max=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, vary_q=False, 
                 color_augment=False, color_augment_strength=0.8, 
                 return_hr=False, resize_hr=False, return_freq=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.inp_size_min = inp_size_min
        self.inp_size_max = inp_size_max
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.color_augment = color_augment
        self.color_augment_strength = color_augment_strength
        self.sample_q = sample_q
        self.vary_q = vary_q
        self.return_hr = return_hr
        self.resize_hr = resize_hr
        self.return_freq = return_freq

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        
        if self.color_augment:
            s = self.color_augment_strength
            color_aug_kwarg = {
                "bright": (random.random() * 2.0) * s,
                "saturation": (random.random() * 2.0)* s,
                "hue": (random.random() - 0.5) * s,
                "gamma": (random.random() * 2.0) * s,
            }
            img = augments.apply_color_distortion(img, **color_aug_kwarg)
        s = random.uniform(self.scale_min, self.scale_max)

        w_lr = round(random.uniform(min(min(self.inp_size_min*s, img.shape[-2]), img.shape[-1]) // s, 
                                    min(min(self.inp_size_max*s, img.shape[-2]), img.shape[-1]) // s))
        w_hr = round(w_lr * s)
        x0 = random.randint(0, img.shape[-2] - w_hr)
        y0 = random.randint(0, img.shape[-1] - w_hr)
        crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        if self.return_freq:
            f_img = tfft.hfft(img.movedim((0,1,2),(2,0,1)), norm="ortho").movedim((0,1,2),(1,2,0))
            f_crop_hr = f_img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            if self.resize_hr:
                if self.inp_size is None:
                    f_crop_hr = resize_fn(f_crop_hr, round(w_lr * s))
                else:
                    f_crop_hr = resize_fn(f_crop_hr, round(self.inp_size * s))
        if self.inp_size is None:
            crop_lr = img[:, x0 + round(w_lr/s): x0 + w_lr + round(w_lr/s), y0 + round(w_lr/s): y0 + w_lr + round(w_lr/s)]
        else:
            crop_lr = resize_fn(img[:, x0 + round(w_lr/s): x0 + w_lr + round(w_lr/s), y0 + round(w_lr/s): y0 + w_lr + round(w_lr/s)], self.inp_size)
        if self.resize_hr:
            if self.inp_size is None:
                crop_hr = resize_fn(crop_hr, round(w_lr * s))
            else:
                crop_hr = resize_fn(crop_hr, round(self.inp_size * s))

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
            if self.return_freq:
                f_crop_hr = augment(f_crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        if self.return_freq:
            hr_freq = to_frequency_samples(f_crop_hr.contiguous())

        if self.sample_q is not None:
            if self.vary_q:
                sample_lst = np.random.choice(len(hr_coord), 
                                              min(round(self.sample_q * s), len(hr_coord)), 
                                              replace=False)
            else:
                sample_lst = np.random.choice(len(hr_coord), 
                                              min(self.sample_q, len(hr_coord)), 
                                              replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
            if self.return_freq:
                hr_freq = hr_freq[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        result = {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
        }
        if self.return_freq:
            result['f_gt'] = hr_freq
        if self.return_hr:
            result["hr"] = crop_hr
        return result

    def shuffle_mapping(self):
        self.dataset.shuffle_mapping()
