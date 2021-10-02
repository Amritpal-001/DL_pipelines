import logging
from typing import List

import numpy as np

from qxr_utils.image import transforms as tsfms

logger = logging.getLogger("preprocess")


def get_zoom_transform(zoom_im_size: int = 512) -> np.ndarray:
    """Function to get chest zoom transforms

    Args:
        arr (np.ndarray): 2D image in nparray format
        zoom_im_size (int): rise to which image should be resized in side the transform

    Raises:
        RuntimeError: If problems in transform function raises runtime error

    Returns:
        np.ndarray: returns the transformed array
    """
    zoom_transform = tsfms.compose([tsfms.resize_hard(zoom_im_size), tsfms.scale, tsfms.clip])
    try:
        return zoom_transform
    except Exception as e:
        logger.exception("Error while getting zoom transforms")
        raise RuntimeError("Error while getting zoom transforms") from e


def zoom_pad(
    zoom_coordinates: np.ndarray, original_im_height: int, original_im_width: int, zoom_im_size: int, padding: int = 10
) -> List[int]:
    """modifying the zoom cordinates by adding 10% padding to the image.
    Padding percentage is adjustable in the args

    Args:
        zoom_coordinates (np.ndarray): 1x4 numpy array of zoom cordinates obtained from the neural network model.
        original_im_height (int): Original Image Height
        original_im_width (int): Original Image Width
        zoom_im_size (int): Zoom transform image size
        padding (int, optional): Percentage extra region needed. Defaults to 10.

    Returns:
        List: zoom coordinates of size 4 which are to be applied to the original image
    """
    # scaling zoom coordinates to original image size
    left = zoom_coordinates[0] * (original_im_width / zoom_im_size)
    top = zoom_coordinates[1] * (original_im_height / zoom_im_size)
    right = zoom_coordinates[2] * (original_im_width / zoom_im_size)
    bottom = zoom_coordinates[3] * (original_im_height / zoom_im_size)
    out = [top, bottom, left, right]
    out = [int(x) for x in out]

    pad = padding / 100
    # Adding padding to the image
    top = max(0.0, out[0] - (out[1] - out[0]) * pad)
    bottom = min(out[1] + (out[1] - out[0]) * pad, original_im_height)
    left = max(0.0, out[2] - (out[3] - out[2]) * pad)
    right = min(out[3] + (out[3] - out[2]) * pad, original_im_width)
    out = [top, bottom, left, right]
    original_zoom_cordinates = [int(x) for x in out]
    return original_zoom_cordinates
