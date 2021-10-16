from typing import List

import numpy as np

import qxr_utils.image.transforms as tsfms


def generate_diaphragm_scan(side_borders: List, im: np.ndarray, im_size: int = 320) -> np.ndarray:
    """Generates diaphragm scan image from full size numpy array and side borders

    Args:
        side_borders (List): list of 2 numpy arrays. each numpy array is a len 4 list
        im (np.ndarray): 2D numpy array image
        im_size (int, optional): output image size. Defaults to 320.

    Returns:
        np.ndarray: diaphragm scan image of size 320x320
    """
    images = []
    half_im_size = im_size // 2
    height, width = np.shape(im)
    for i in range(len(side_borders)):
        side_border = side_borders[i]
        top, bot, left, right = side_border[0:4]
        bot = min(int(bot * 1.136), height)
        top = top + (bot - top) // 2
        if i == 1:
            left = max(0, int(right - (right - left) * 1.05))
        if i == 0:
            right = min(int(left + (right - left) * 1.05), width)
        im_copy = np.copy(im)
        im_copy = im_copy[(slice(top, bot), slice(left, right))]
        im_copy = tsfms.resize(im_size, half_im_size)(im_copy)
        images.append(im_copy)
    diaphragm_image = np.zeros((im_size, im_size))
    diaphragm_image[:, :half_im_size] = images[1]
    diaphragm_image[:, half_im_size:] = images[0]
    diaphragm_image = tsfms.compose([tsfms.scale, tsfms.clip])(diaphragm_image)
    return diaphragm_image
