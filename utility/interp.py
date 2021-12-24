import math
from tqdm import tqdm
from torch import Tensor
from typing import Tuple, List


def get_color_bil(x: float, y: float, image: Tensor):
    """
    Selects a color from an image using bilinear interpolation
    """
    i, j = math.floor(y.item()), math.floor(x.item())
    c0 = image[max(i, 0)][max(j, 0)]
    c1 = image[max(i, 0)][min(j+1, image.shape[1] - 1)]
    c2 = image[min(i+1, image.shape[0] - 1)][min(j+1, image.shape[1] - 1)]
    c3 = image[min(i+1, image.shape[0] - 1)][max(j, 0)]
    w_top = y - i
    w_left = x - j
    
    # Step 1: interpolate along x-axis
    color_top = c0 * (1 - w_left) + c1 * w_left
    color_bot = c3 * (1 - w_left) + c2 * w_left
    
    # Step 2: interpolate along y-axis
    color = color_top * (1 - w_top) + w_top * color_bot

    return color


def get_color_nearest(x: float, y: float, image: Tensor):
    i = min(max(round(y.item()), 0), image.shape[0] - 1)
    j = min(max(round(x.item()), 0), image.shape[1] - 1)
    
    return image[i][j]


def compute_gaussian_density(x: Tuple[float, float], mean: Tuple[float, float], std: Tuple[float, float]):
    exp_term = torch.exp(-0.5 * (((x[0] - mean[0]) / std[0]) ** 2 + ((x[1] - mean[1]) / std[1]) ** 2))
    std_term = 1 / (std[0] * std[1])
    const_term = 1 / (2 * np.pi)
    
    return exp_term * std_term * const_term


def gaussian_interpolation(means: Tensor, stds: Tensor, img: Tensor, radius: int):
    """
    Performs a gaussian process interpolation
    means: [num_coords, 2]
    stds: [num_coords, 2]
    images: [image_width, image_height]
    density_threshold --- we do not 
    """
    result = torch.zeros_like(img)
    total_weights = torch.zeros(img.shape[0], img.shape[1])

    for i in tqdm(range(0, len(means))):
        center = (round(means[i][0].item()), round(means[i][1].item()))
        color = get_color_nearest(means[i][0], means[i][1], img)

        for y_shift in range(-radius, radius + 1):
            for x_shift in range(-radius, radius + 1):
                pixel_pos = (center[0] + x_shift, center[1] + y_shift)
                
                if (pixel_pos[0] < 0 or pixel_pos[0] >= img.shape[1]) or (pixel_pos[1] < 0 or pixel_pos[1] >= img.shape[0]):
                    continue

                weight = compute_gaussian_density(pixel_pos, means[i], stds[i])
                #weight = means[i][0] + means[i][1]
                #weight = (1 / ((pixel_pos[0] - means[i][0]) ** 2 + (pixel_pos[1] - means[i][1]) ** 2 + 1e-10)) ** 1
                if weight > 0:
                    result[pixel_pos[1], pixel_pos[0]] += color * weight
                    total_weights[pixel_pos[1], pixel_pos[0]] += weight
    
    # Now, we should normalize the colors by the total weight
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if total_weights[i][j] > 0:
                result[i][j] /= total_weights[i][j]

    return result, total_weights