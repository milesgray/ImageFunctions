# https://github.com/EndlessSora/focal-frequency-loss/blob/master/metrics/metric_utils.py#L254
import torch
import torch.nn as nn
import numpy as np

from .registry import register

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
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    img = img.astype(np.float64)
    return img

def lfd(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate LFD (Log Frequency Distance).
    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the LFD calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
    Returns:
        float: lfd result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    img1 = img1.transpose(2, 0, 1)
    img2 = img2.transpose(2, 0, 1)
    freq1 = np.fft.fft2(img1)
    freq2 = np.fft.fft2(img2)
    return np.log(np.mean((freq1.real - freq2.real)**2 + (freq1.imag - freq2.imag)**2) + 1.0)

@register("lfd")
class LFDMetric(nn.Module):
    def __init__(self, 
                 crop_border=0, 
                 input_order='HWC', ):
        """Calculate LFD (Log Frequency Distance).
        Ref:
        Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
        <https://arxiv.org/pdf/2012.12821.pdf>
        Args:
            img1 (ndarray): Images with range [0, 255].
            img2 (ndarray): Images with range [0, 255].
            crop_border (int): Cropped pixels in each edges of an image. These
                pixels are not involved in the LFD calculation. Default: 0.
            input_order (str): Whether the input order is 'HWC' or 'CHW'.
                Default: 'HWC'.
        Returns:
            float: lfd result.
        """
        super().__init__()
        self.crop_border = crop_border
        self.input_order = input_order
        
    def forward(self, x, y):
        result = lfd(x, y,                  
                     crop_border=self.crop_border,
                     input_order=self.input_order) 
)