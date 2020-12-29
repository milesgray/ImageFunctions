import os
import time
import shutil
import math

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter
import piq

class Averager():
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v

class Timer():
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v

def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)

_log_path = None

def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)

def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


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

def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)

def rgb2ycbcrT(rgb):
    rgb = rgb.numpy().transpose(1, 2, 0)
    yCbCr = sc.rgb2ycbcr(rgb) / 255

    return torch.Tensor(yCbCr[:, :, 0])

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_SSIM(input, target, rgb_range, shave):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''

    c, h, w = input.size()
    if c > 1:
        input = input.mul(255).clamp(0, 255).round()
        target = target[:, 0:h, 0:w].mul(255).clamp(0, 255).round()
        input = rgb2ycbcrT(input)
        target = rgb2ycbcrT(target)
    else:
        input = input
        target = target[:, 0:h, 0:w]
    input = input[shave:(h - shave), shave:(w - shave)]
    target = target[shave:(h - shave), shave:(w - shave)]
    return ssim(input.numpy(), target.numpy())
