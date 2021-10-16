from typing import Callable, List

import cv2
import numpy as np

import qxr_utils.image.transforms as tsfms


def flip_transform_image(arr: np.ndarray, angle: int, flip: int) -> np.ndarray:
    """Transforms an image based on the output from flip rot model

    Args:
        arr (np.ndarray): input image 2D numpy array
        angle (int): angle at which rotation should happen. choice of (0, 90, 180, 270)
        flip (int): to flip or to not flip. choice of (0,1)

    Returns:
        np.ndarray: flipped+rotated image
    """
    if angle != 0:
        h, w = arr.shape[0], arr.shape[1]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        arr = cv2.warpAffine(arr, M, (h, w))
    if flip == 1:
        arr = np.fliplr(arr)

    return arr


def fliprot_transform():
    transform_list: List[Callable] = []
    transform_list.append(tsfms.lnorm)
    return tsfms.compose(transform_list)
