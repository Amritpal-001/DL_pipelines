import numpy as np

from qxr_utils.preprocess import downsample


def invert_transforms():
    invert_transform = downsample.get_downsample_transform(im_size=224, use_rmblack=False)
    return invert_transform
