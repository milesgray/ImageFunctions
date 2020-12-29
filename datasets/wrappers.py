import functools
import random
import math
from PIL import Image

import numpy as np
import torch
import torch.fft as tfft
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples, to_frequency_samples, resize_fn
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

@register('sr-explicit-downsampled-randcrop')
class SRExplicitDownsampledRandCrop(Dataset):
    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, color_augment=False, color_augment_strength=0.8,
                 return_hr=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.color_augment = color_augment
        self.color_augment_strength = color_augment_strength
        self.return_hr = return_hr

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

        result = {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
        if self.return_hr:
            result["hr"] = crop_hr
        return result

@register('sr-randrange-downsampled-randcrop')
class SRRandRangeDownsampledRandCrop(Dataset):
    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, vary_q=False, color_augment=False, color_augment_strength=0.8, 
                 return_hr=False):
        self.dataset = dataset
        self.inp_size = inp_size
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

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = round(random.uniform(min(min(self.inp_size*s, img.shape[-2] // s), img.shape[-1] // s) // s, 
                                        min(min(self.inp_size*s*s, img.shape[-2] // s), img.shape[-1] // s) // s))
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
            if self.vary_q:
                sample_lst = np.random.choice(len(hr_coord), min(round(self.sample_q * s), len(hr_coord)), replace=False)
            else:
                sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        result = {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
        if self.return_hr:
            result["hr"] = crop_hr
        return result

@register('sr-setrange-downsampled-randcrop')
class SRSetRangeDownsampledRandCrop(Dataset):
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

@register('sr-randrange-downsampled-randcrop')
class SRRandRangeDownsampledRandCrop(Dataset):
    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, vary_q=False, color_augment=False, color_augment_strength=0.8, 
                 return_hr=False):
        self.dataset = dataset
        self.inp_size = inp_size
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

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = round(random.uniform(min(min(self.inp_size*s, img.shape[-2] // s), img.shape[-1] // s) // s, 
                                        min(min(self.inp_size*s*s, img.shape[-2] // s), img.shape[-1] // s) // s))
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
            if self.vary_q:
                sample_lst = np.random.choice(len(hr_coord), min(round(self.sample_q * s), len(hr_coord)), replace=False)
            else:
                sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        result = {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
        if self.return_hr:
            result["hr"] = crop_hr
        return result

@register('zr-setrange-downsampled-randcrop')
class ZRSetRangeDownsampledRandCrop(Dataset):
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
            crop_lr = img[:, x0 + round(w_lr/s): x0 + w_lr, y0 + round(w_lr/s): y0 + w_lr]
        else:
            crop_lr = resize_fn(img[:, x0 + round(w_lr/s): x0 + w_lr, y0 + round(w_lr/s): y0 + w_lr], self.inp_size)
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
