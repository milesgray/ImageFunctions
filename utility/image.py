import math
from typing import Union, Dict

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import utility as utils
import utility.color as color

class Resolution:
    """Class representing the width and height of an image."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
    
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

    def divisible(self, divisor: int) -> "Resolution":        
        """Trims this resolution to make divisible by a divisor.
        Args:
            divisor (int): The desired number to be divisible by
        Returns:
            a resolution with trimmed to be divisible by divisor
        """
        height = self.height - self.height % divisor
        width = self.width - self.width % divisor
        return Resolution(width, height)

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


def make_divisible(image: np.ndarray, divisor: int) -> np.ndarray:
    """Trim the image if not divisible by the divisor."""
    height, width = image.shape[:2]
    if height % divisor == 0 and width % divisor == 0:
        return image

    new_height = height - height % divisor
    new_width = width - width % divisor

    return image[:new_height, :new_width]

def variance_of_laplacian(image: np.ndarray) -> np.ndarray:
    """Compute the variance of the Laplacian which measure the focus."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CVX_64F).var()

def image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert the image to a uint8 array."""
    if image.dtype == np.uint8:
        return image
    if not issubclass(image.dtype.type, np.floating):
        raise ValueError(
                f'Input image should be a floating type but is of type {image.dtype!r}')
    return (image * UINT8_MAX).clip(0.0, UINT8_MAX).astype(np.uint8)


def image_to_uint16(image: np.ndarray) -> np.ndarray:
    """Convert the image to a uint16 array."""
    if image.dtype == np.uint16:
        return image
    if not issubclass(image.dtype.type, np.floating):
        raise ValueError(
                f'Input image should be a floating type but is of type {image.dtype!r}')
    return (image * UINT16_MAX).clip(0.0, UINT16_MAX).astype(np.uint16)


def image_to_float32(image: np.ndarray) -> np.ndarray:
    """Convert the image to a float32 array and scale values appropriately."""
    if image.dtype == np.float32:
        return image

    dtype = image.dtype
    image = image.astype(np.float32)
    if dtype == np.uint8:
        return image / UINT8_MAX
    elif dtype == np.uint16:
        return image / UINT16_MAX
    elif dtype == np.float64:
        return image
    elif dtype == np.float16:
        return image

    raise ValueError(f'Not sure how to handle dtype {dtype}')



def checkerboard(h, w, size=8, true_val=1.0, false_val=0.0):
    """Creates a checkerboard pattern with height h and width w."""
    i = int(math.ceil(h / (size * 2)))
    j = int(math.ceil(w / (size * 2)))
    pattern = np.kron([[1, 0] * j, [0, 1] * j] * i,
                                        np.ones((size, size)))[:h, :w]

    true = np.full_like(pattern, fill_value=true_val)
    false = np.full_like(pattern, fill_value=false_val)
    return np.where(pattern > 0, true, false)


def pad_image(image, pad=0, pad_mode='constant', pad_value=0.0):
    """Pads a batched image array."""
    batch_shape = image.shape[:-3]
    padding = [
            *[(0, 0) for _ in batch_shape],
            (pad, pad), (pad, pad), (0, 0),
    ]
    if pad_mode == 'constant':
        return np.pad(image, padding, pad_mode, constant_values=pad_value)
    else:
        return np.pad(image, padding, pad_mode)


def split_tiles(image, tile_size):
    """Splits the image into tiles of size `tile_size`."""
    # The copy is necessary due to the use of the memory layout.
    if image.ndim == 2:
        image = image[..., None]
    image = np.array(image)
    image = make_divisible(image, tile_size).copy()
    height = width = tile_size
    nrows, ncols, depth = image.shape
    stride = image.strides

    nrows, m = divmod(nrows, height)
    ncols, n = divmod(ncols, width)
    if m != 0 or n != 0:
        raise ValueError('Image must be divisible by tile size.')

    return np.lib.stride_tricks.as_strided(
            np.ravel(image),
            shape=(nrows, ncols, height, width, depth),
            strides=(height * stride[0], width * stride[1], *stride),
            writeable=False)


def join_tiles(tiles):
    """Reconstructs the image from tiles."""
    return np.concatenate(np.concatenate(tiles, 1), 1)


def make_grid(batch, grid_height=None, zoom=1, old_buffer=None, border_size=1):
    """Creates a grid out an image batch.
    Args:
        batch: numpy array of shape [batch_size, height, width, n_channels]. The
            data can either be float in [0, 1] or int in [0, 255]. If the data has
            only 1 channel it will be converted to a grey 3 channel image.
        grid_height: optional int, number of rows to have. If not given, it is
            set so that the output is a square. If -1, then tiling will only be
            vertical.
        zoom: optional int, how much to zoom the input. Default is no zoom.
        old_buffer: Buffer to write grid into if possible. If not set, or if shape
            doesn't match, we create a new buffer.
        border_size: int specifying the white spacing between the images.
    Returns:
        A numpy array corresponding to the full grid, with 3 channels and values
        in the [0, 255] range.
    Raises:
        ValueError: if the n_channels is not one of [1, 3].
    """

    batch_size, height, width, n_channels = batch.shape

    if grid_height is None:
        n = int(math.ceil(math.sqrt(batch_size)))
        grid_height = n
        grid_width = n
    elif grid_height == -1:
        grid_height = batch_size
        grid_width = 1
    else:
        grid_width = int(math.ceil(batch_size/grid_height))

    if n_channels == 1:
        batch = np.tile(batch, (1, 1, 1, 3))
        n_channels = 3

    if n_channels != 3:
        raise ValueError('Image batch must have either 1 or 3 channels, but '
                                         'was {}'.format(n_channels))

    # We create the numpy buffer if we don't have an old buffer or if the size has
    # changed.
    shape = (height * grid_height + border_size * (grid_height - 1),
                     width * grid_width + border_size * (grid_width - 1),
                     n_channels)
    if old_buffer is not None and old_buffer.shape == shape:
        buf = old_buffer
    else:
        buf = np.full(shape, 255, dtype=np.uint8)

    multiplier = 1 if np.issubdtype(batch.dtype, np.integer) else 255

    for k in range(batch_size):                                       
        i = k // grid_width
        j = k % grid_width
        arr = batch[k]
        x, y = i * (height + border_size), j * (width + border_size)
        buf[x:x + height, y:y + width, :] = np.clip(multiplier * arr,
                                                                                                0, 255).astype(np.uint8)

    if zoom > 1:
        buf = buf.repeat(zoom, axis=0).repeat(zoom, axis=1)
    return buf