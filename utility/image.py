import math
from typing import Union, Dict

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import utility as util

def make_img_coeff(data_norm):
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    try:
        result = data_norm.copy()
        result = util.dict_apply(result,
                            lambda x: util.dict_apply(x,
                                                 lambda y: torch.FloatTensor(y))
        )
        result['inp'] = util.dict_apply(result['inp'],
                                   lambda x: x.view(1, -1, 1, 1))
        result['gt'] = util.dict_apply(result['gt'],
                                  lambda x: x.view(1, 1, -1))

        if torch.cuda.is_available():
            result = util.dict_apply(result,
                                lambda x: util.dict_apply(x,
                                                 lambda y: y.cuda())
            )
        return result
    except Exception as e:
        print(f"Img coeff fail:\n{e}")
        return data_norm.copy()

def reshape(pred, target_shape):
    ih, iw = target_shape
    s = math.sqrt(pred.shape[1] / (ih * iw))
    shape = [pred.shape[0], round(ih * s), round(iw * s), 3]
    pred = pred.view(*shape) \
        .permute(0, 3, 1, 2).contiguous()
    return pred

def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.
    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.
    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.
    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))

def to_frequency_samples(f_img):
    freq = f_img.view(4, -1).permute(1, 0)
    return freq

def to_y_channel(img):
    """Change to Y channel of YCbCr.
    Args:
        img (ndarray): Images with range [0, 255].
    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.

# ----
# COLOR SPACES

def rgb2ycbcrT(rgb):
    rgb = rgb.numpy().transpose(1, 2, 0)
    yCbCr = sc.rgb2ycbcr(rgb) / 255

    return torch.Tensor(yCbCr[:, :, 0])

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    img_type = img.dtype
    img.astype(np.float32)
    if img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(img_type)

def rgb2lab(img, mean_cent=False):
    from skimage import color
    img_lab = color.rgb2lab(img)
    if(mean_cent):
        img_lab[:,:,0] = img_lab[:,:,0]-50
    return img_lab

def rgb2yiq(x: torch.Tensor) -> torch.Tensor:
    r"""Convert a batch of RGB images to a batch of YIQ images
    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.
    Returns:
        Batch of images with shape (N, 3, H, W). YIQ colour space.
    """
    yiq_weights = torch.tensor([
        [0.299, 0.587, 0.114],
        [0.5959, -0.2746, -0.3213],
        [0.2115, -0.5227, 0.3112]]).t().to(x)
    x_yiq = torch.matmul(x.permute(0, 2, 3, 1), yiq_weights).permute(0, 3, 1, 2)
    return x_yiq


def rgb2lhm(x: torch.Tensor) -> torch.Tensor:
    r"""Convert a batch of RGB images to a batch of LHM images
    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.
    Returns:
        Batch of images with shape (N, 3, H, W). LHM colour space.
    Reference:
        https://arxiv.org/pdf/1608.07433.pdf
    """
    lhm_weights = torch.tensor([
        [0.2989, 0.587, 0.114],
        [0.3, 0.04, -0.35],
        [0.34, -0.6, 0.17]]).t().to(x)
    x_lhm = torch.matmul(x.permute(0, 2, 3, 1), lhm_weights).permute(0, 3, 1, 2)
    return x_lhm


def xyz2lab(x: torch.Tensor, illuminant: str = 'D50', observer: str = '2') -> torch.Tensor:
    r"""Convert a batch of XYZ images to a batch of LAB images
    Args:
        x: Batch of images with shape (N, 3, H, W). XYZ colour space.
        illuminant: {“A”, “D50”, “D55”, “D65”, “D75”, “E”}, optional. The name of the illuminant.
        observer: {“2”, “10”}, optional. The aperture angle of the observer.
    Returns:
        Batch of images with shape (N, 3, H, W). LAB colour space.
    """
    epsilon = 0.008856
    kappa = 903.3
    illuminants: Dict[str, Dict] = \
        {"A": {'2': (1.098466069456375, 1, 0.3558228003436005),
               '10': (1.111420406956693, 1, 0.3519978321919493)},
         "D50": {'2': (0.9642119944211994, 1, 0.8251882845188288),
                 '10': (0.9672062750333777, 1, 0.8142801513128616)},
         "D55": {'2': (0.956797052643698, 1, 0.9214805860173273),
                 '10': (0.9579665682254781, 1, 0.9092525159847462)},
         "D65": {'2': (0.95047, 1., 1.08883),  # This was: `lab_ref_white`
                 '10': (0.94809667673716, 1, 1.0730513595166162)},
         "D75": {'2': (0.9497220898840717, 1, 1.226393520724154),
                 '10': (0.9441713925645873, 1, 1.2064272211720228)},
         "E": {'2': (1.0, 1.0, 1.0),
               '10': (1.0, 1.0, 1.0)}}

    illuminants_to_use = torch.tensor(illuminants[illuminant][observer]).to(x).view(1, 3, 1, 1)

    tmp = x / illuminants_to_use

    mask_below = tmp <= epsilon
    mask_above = tmp > epsilon
    tmp = torch.pow(tmp, 1. / 3.) * mask_above + (kappa * tmp + 16.) / 116. * mask_below

    weights_xyz_to_lab = torch.tensor([[0, 116., 0],
                                       [500., -500., 0],
                                       [0, 200., -200.]]).to(x)
    bias_xyz_to_lab = torch.tensor([-16., 0., 0.]).to(x).view(1, 3, 1, 1)

    x_lab = torch.matmul(tmp.permute(0, 2, 3, 1), weights_xyz_to_lab.t()).permute(0, 3, 1, 2) + bias_xyz_to_lab
    return x_lab

def rgb2xyz(x: torch.Tensor) -> torch.Tensor:
    r"""Convert a batch of RGB images to a batch of XYZ images
    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.
    Returns:
        Batch of images with shape (N, 3, H, W). XYZ colour space.
    """
    mask_below = (x <= 0.04045).to(x)
    mask_above = (x > 0.04045).to(x)

    tmp = x / 12.92 * mask_below + torch.pow((x + 0.055) / 1.055, 2.4) * mask_above

    weights_rgb_to_xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                                       [0.2126729, 0.7151522, 0.0721750],
                                       [0.0193339, 0.1191920, 0.9503041]]).to(x)

    x_xyz = torch.matmul(tmp.permute(0, 2, 3, 1), weights_rgb_to_xyz.t()).permute(0, 3, 1, 2)
    return x_xyz

def rgb2lmn(x: torch.Tensor) -> torch.Tensor:
    r"""Convert a batch of RGB images to a batch of LMN images
    Args:
        x: Batch of images with shape (N, 3, H, W). RGB colour space.
    Returns:
        Batch of images with shape (N, 3, H, W). LMN colour space.
    """
    weights_rgb_to_lmn = torch.tensor([[0.06, 0.63, 0.27],
                                       [0.30, 0.04, -0.35],
                                       [0.34, -0.6, 0.17]]).t().to(x)
    x_lmn = torch.matmul(x.permute(0, 2, 3, 1), weights_rgb_to_lmn).permute(0, 3, 1, 2)
    return x_lmn
