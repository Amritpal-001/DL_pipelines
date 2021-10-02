import logging
from typing import Callable, List, Optional, Union

import numpy as np

import qxr_utils.image.transforms as tsfms

logger = logging.getLogger("preprocess")


def get_downsample_image(
    im: np.ndarray, im_size: int = 224, zoom_coords: Optional[Union[np.ndarray, List]] = None, use_rmblack: bool = True
) -> np.ndarray:
    """Downsamples an image with default transforms which work for qxrtesting, cxr_product and framework

    Args:
        im (np.ndarray): 2D numpy array image
        im_size (int, optional): target image size to which we should downscale. Defaults to 224.
        zoom_coords (Optional[Union[np.ndarray, List]], optional): zoom coordinates to crop into. Defaults to None.
        use_rmblack (bool, optional): if we should remove black borders. Defaults to True.

    Raises:
        RuntimeError: Raises a runtime error if function breaks in execution

    Returns:
        np.ndarray: Final downsampled 2D NxN array where N is im_size
    """
    try:
        if zoom_coords is not None and len(zoom_coords) == 4:
            im = im[zoom_coords[0] : zoom_coords[1], zoom_coords[2] : zoom_coords[3]]
        return get_downsample_transform(im_size, use_rmblack)(im)
        # return downsample_transform(im)
    except Exception as e:
        logger.exception("Error while generating downsampled array")
        raise RuntimeError("Error while generating downsampled array") from e


def get_downsample_transform(im_size: int = 224, use_rmblack: bool = True):
    transform_list: List[Callable] = []
    if use_rmblack:
        transform_list.append(tsfms.rmblack)
    transform_list.append(tsfms.scale)
    transform_list.append(tsfms.resize_hard(im_size))
    transform_list.append(tsfms.clip)
    return tsfms.compose(transform_list)
