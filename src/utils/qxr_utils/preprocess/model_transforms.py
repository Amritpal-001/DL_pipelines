from typing import Callable

import qxr_utils.image.transforms as tsfms


def get_val_side_model_transform(im_size: int = 320) -> Callable:
    """side model transform, this needs to be applied before passing to a model which is side based.

    Args:
        im_size (int, optional): image width. Image size will be 3im_sizexim_size. Defaults to 320.

    Returns:
        Callable: [description]
    """
    side_transform = tsfms.compose([tsfms.scale, tsfms.resize_hard_3x(im_size), tsfms.clip])
    return side_transform


def get_val_patch_model_trasform(im_size: int = 224) -> Callable:
    """transforms to be given to any model which is patch based

    Args:
        im_size (int, optional): [description]. Defaults to 224.

    Returns:
        Callable: [description]
    """
    patch_transform = tsfms.compose([tsfms.scale, tsfms.resize_hard(im_size), tsfms.clip])
    return patch_transform
