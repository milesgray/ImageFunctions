import math
from typing import Union, Dict

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import utility as utils
import utility.color as color

class Resolution(NamedTuple("Resolution", [("width", int), ("height", int)])):
    """Class representing the width and height of an image."""
    def scale_to_height(self, height: int) -> "Resolution":
        """Scales this resolution while maintaining the aspect ratio.
        Args:
            height (int): The desired new height
        Returns:
            a resolution with the specified height but the same aspect ratio
        """
        width = self.width * height // self.height
        return Resolution(width, height)

    def square(self) -> "Resolution":
        """Returns a square version of this resolution."""
        size = min(self.width, self.height)
        return Resolution(size, size)

class ImageWrap:
    def __init__(self, img, space="bgr"):
        self.img = img
        self.space = space

    def reorder(self, input_order='HWC'):
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
        if len(self.img.shape) == 2:
            self.img = self.img[..., None]
        if input_order == 'CHW':
            self.img = self.img.transpose(1, 2, 0)
        return self.img
    
    def reshape(self, target_shape):
        ih, iw = target_shape
        s = math.sqrt(self.img.shape[1] / (ih * iw))
        shape = [self.img.shape[0], round(ih * s), round(iw * s), 3]
        self.img = self.img.view(*shape) \
            .permute(0, 3, 1, 2).contiguous()
        return self.img

def calc_dataset_stats(train_data):
    dataset_size = len(train_data.targets)
    total = {"R": 0, "G": 0, "B": 0}
    total_pixel = 0
    for i, batch in enumerate(tqdm(train_data)):    
        for img in batch["images"]:        
            total_pixel = total_pixel + img.shape[1] * img.shape[2]

            total["R"] = total["R"] + torch.sum((img[0, :, :]))
            total["G"] = total["G"] + torch.sum((img[1, :, :]))
            total["B"] = total["B"] + torch.sum((img[2, :, :]))
        if i > len(train_data):
            break
    data_stats["mean"][0] = total["R"]/total_pixel
    data_stats["mean"][1] = total["G"]/total_pixel
    data_stats["mean"][2] = total["B"]/total_pixel
    for i, batch in enumerate(tqdm(train_data)):
        imgs = batch["images"]
        for img in imgs:
            total["R"] = total["R"] + torch.sum((img[0, :, :] - data_stats["mean"][0]) ** 2)
            total["G"] = total["G"] + torch.sum((img[1, :, :] - data_stats["mean"][1]) ** 2)
            total["B"] = total["B"] + torch.sum((img[2, :, :] - data_stats["mean"][2]) ** 2)
        if i > len(train_data):
            break
    data_stats["std"][0] = torch.sqrt(total["R"] / total_pixel)
    data_stats["std"][1]= torch.sqrt(total["G"] / total_pixel)
    data_stats["std"][2] = torch.sqrt(total["B"] / total_pixel)

    print(f'\nmeans:\n{data_stats["mean"]},std:\n{data_stats["std"]}')
    return data_stats

def make_img_coeff(data_norm):
    if data_norm is None:
        data_norm = {
            'inp': {'sub': 0, 'div': 1},
            'gt': {'sub': 0, 'div': 1}
        }
    try:
        result = data_norm.copy()
        result = utils.dict_apply(result,
                            lambda x: utils.dict_apply(x,
                                                 lambda y: torch.FloatTensor(y))
        )
        result['inp'] = utils.dict_apply(result['inp'],
                                   lambda x: x.view(1, -1, 1, 1))
        result['gt'] = utils.dict_apply(result['gt'],
                                  lambda x: x.view(1, 1, -1))

        if torch.cuda.is_available():
            result = utils.dict_apply(result,
                                lambda x: utils.dict_apply(x,
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