import gzip
from random import randint
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import find_objects as sp_find_objects
from scipy.ndimage import label as sp_label
from skimage import filters


def np_compress(original_mask: np.ndarray) -> Mapping[str, Any]:
    """
    Compresses a numpy array and returns it as a dict
    Args:
        original_mask: numpy array

    Returns:
        A dictionary of the form
        {
            "compressed_mask": bytes,
            "mask_shape": shape of the array,
            "dtype": numpy.dtype,
        }

    """
    orig_shape = np.shape(original_mask)
    np_dtype = original_mask.dtype
    flattened_mask = original_mask.flatten()
    byte_mask = flattened_mask.tobytes()
    compressed_mask = gzip.compress(byte_mask)
    comp_dict = {
        "compressed_mask": compressed_mask,
        "mask_shape": orig_shape,
        "dtype": np_dtype,
    }
    return comp_dict


def np_decompress(comp_dict: Mapping[str, Any]) -> np.ndarray:
    """
    decompresses a compressed numpy array
    Args:
        comp_dict: output of np_compress, a dictionary of form
        {'compressed_mask':bytes, 'mask_shape':tuple, 'dtype': numpy.dtype}

    Returns:
        uncompressed mask of shape comp_dict['mask_shape']

    """

    byte_mask = gzip.decompress(comp_dict["compressed_mask"])
    flattened_mask = np.frombuffer(byte_mask, dtype=comp_dict["dtype"])
    orig_mask = flattened_mask.reshape(comp_dict["mask_shape"])
    return orig_mask


def type_check(im: Union[np.ndarray, dict]) -> np.ndarray:
    """
    Returns numpy array if the input is a dict
    Args:
        im: numpy array or a dict

    Returns:
        numpy array

    """
    if isinstance(im, np.ndarray):
        return im
    elif isinstance(im, dict):
        return im["input"]


def give_shape_2dslice(inp: Tuple[slice, slice]) -> Tuple[int, int]:
    x1 = inp[0].start
    x2 = inp[0].stop
    y1 = inp[1].start
    y2 = inp[1].stop
    return ((x2 - x1), (y2 - y1))


def scale(arr: np.ndarray) -> np.ndarray:
    """
    scales all values of numpy array to [0,1]
    Args:
        arr: numpy array

    Returns:
        scaled array

    """
    # TODO assertion on type of numpy array ?
    eps = 1e-10
    if arr.dtype in [np.float64, np.float32, np.float16]:
        eps = np.finfo(arr.dtype).eps
    arr = arr - arr.min()
    arr = arr / (arr.max() + eps)
    return arr


def cast_uint8(im: np.ndarray) -> np.ndarray:
    """
    casts a numpy array to unsigned int dtype. Useful for cv2 functions
    since they consume uint8 arrays
    Args:
        im: numpy array

    Returns:
        same numpy array as unsigned int dtype

    """
    # TODO assetion of dtype?
    if im.dtype == bool:
        im = im.astype(int)

    im = scale(im)
    im = im * 255
    im = im.astype(dtype=np.uint8)
    return im


def clip(arr: np.ndarray) -> np.ndarray:
    """
    Clips the array between 0 and 1
    Args:
        arr: numpy array

    Returns:
        clipped numpy array

    """
    arr = np.clip(arr, a_max=1.0, a_min=0)
    return arr


def _good_return(im: Union[np.ndarray, dict], out: np.ndarray) -> Optional[Union[np.ndarray, dict]]:
    """
    returns 'out' as np.ndarray if the input 'im' is an ndarray
    inserts out into im['input'] if the input 'im' is a dict
    Args:
        im: np.ndarray or dict
        out: np.ndarray

    Returns:
        returns the same type as input

    """
    if isinstance(im, np.ndarray):
        return out
    elif isinstance(im, dict):
        im["input"] = out
        return im
    return None


def squeeze(inp: Union[np.ndarray, dict], th1: float = 0, th2: float = 0) -> Union[np.ndarray, dict]:
    """
    # To be removed
    Squeezes the values of an image that lie between the two given
    thresholds. the pixel values > th2 and the pixel values < th1 are multiplied
    by constants and the final image is scaled
    Args:
        inp: numpy array
        th1: threshold 1
        th2: threshold 2

    Returns:
        squeezed array
    """
    im = type_check(inp)
    if th1 == 0 and th2 == 0:
        th1 = 0.1 * randint(0, 4)
        th2 = 0.1 * randint(5, 10)
    gtarr = (im >= th2) * im
    ltarr = (im < th1) * im
    nim = (im > th1) * (im < th2) * im
    new_im = scale(1.2 * ltarr + 0.9 * gtarr + nim)
    return _good_return(inp, scale(new_im))


def sparse(inp: Union[np.ndarray, dict], th1: float = 0, th2: float = 0) -> Union[np.ndarray, dict]:
    """
    # To be removed
    Opposite of squeeze, stretches out the pixel values that lie between
    the two given thresholds
    Args:
        inp: np.ndarray or dict
        th1: lower threshold
        th2: higher threshold

    Returns:
       stretched out image

    """
    im = type_check(inp)
    if th1 == 0 and th2 == 0:
        th1 = 0.1 * randint(0, 4)
        th2 = 0.1 * randint(5, 10)
    gtarr = (im >= th2) * im
    ltarr = (im < th1) * im
    nim = (im > th1) * (im < th2) * im
    nim = range_scale(nim, 0.9 * th1, 1.2 * th2)
    new_im = scale(0.9 * ltarr + 1.2 * gtarr + nim)
    return _good_return(inp, scale(new_im))


def range_scale(im: np.ndarray, th1: float = 0, th2: float = 0) -> np.ndarray:
    """
    # To be removed
    Scale the image between the two given thresholds
    Args:
        im: np.ndarray or dict
        th1: lower threshold
        th2: higher threshold

    Returns:
        scaled image

    """
    non_zero_min: int = ((im == 0) * 1 + im).min()
    dif = th2 - th1
    im2 = im - non_zero_min
    im2 = (im2 > 0) * im2
    im2 = im2 * (dif / (im2.max() + 0.00001))
    im2 = im2 + th1
    im2 = (im2 > th1) * im2
    return im2


def get_borders(im: np.ndarray, th: float) -> Tuple[int, int, int, int]:
    """
    Returns borders of an image by thresholding each strip. Used in rmblack function
    #TODO vectorize the operations, currently this takes too much time
    Args:
        im: numpy array
        th: threshold

    Returns:
        left, right, up, down borders

    """
    x = im.shape[0]
    y = im.shape[1]
    mn = im.mean()
    bl, br, bu, bd = (0, x, 0, y)
    for i in range(0, x):
        strip = im[i : i + 1, :]
        if strip.std() > th or strip.mean() > mn / 2:
            bl = i
            break
    for i in range(x, 0, -1):
        strip = im[i - 1 : i, :]
        if strip.std() > th or strip.mean() > mn / 2:
            br = i
            break
    for i in range(0, y):
        strip = np.transpose(im[:, i : i + 1], (1, 0))
        if strip.std() > th or strip.mean() > mn / 2:
            bu = i
            break
    for i in range(y, 0, -1):
        strip = np.transpose(im[:, i - 1 : i], (1, 0))
        if strip.std() > th or strip.mean() > mn / 2:
            bd = i
            break
    return bl, br, bu, bd


def get_indices_from_2dslice(
    slice_2d: Tuple[slice, slice], start_border: Optional[Union[np.ndarray, List]] = None
) -> np.ndarray:
    """Return a numpy array from 2d slice

    Args:
        slice_2d (slice): input 2d slice of the shape slice(slice(a,b), slice(c,d))
        start_border (Optional[Union[np.ndarray, List]], optional): 1d numpy array or a list of 4 integers, at which the border counting should start. Defaults to None.

    Returns:
        np.ndarray: output indices in numpy array of shape 1x4
    """
    x1 = slice_2d[0].start
    x2 = slice_2d[0].stop
    y1 = slice_2d[1].start
    y2 = slice_2d[1].stop
    if start_border is not None:
        x1 += start_border[0]
        x2 += start_border[0]
        y1 += start_border[2]
        y2 += start_border[2]
    return np.array([x1, x2, y1, y2])


def get_borders_with_scipy(mask: np.ndarray, get_largest: int = 1) -> Optional[np.ndarray]:
    """Returns borders using scipy's find objects function. Can be used to tighten the bounds of existing mask

    Args:
        mask (np.ndarray): 2D numpy array image
        get_largest (int, optional): 0/1 weather to give largest object among all the borders. Defaults to 1.

    Raises:
        RuntimeError: Returns runtime error if code failes to return a border.

    Returns:
        Optional[np.ndarray]:  1x4 numpy array indicating the tight border crop. If it can't find a border will return None
    """
    try:
        mask = sp_label(mask)[0]
        all_objects = sp_find_objects(mask)
        borders_list = []
        sizelis = []
        for obj in all_objects:
            borders = get_indices_from_2dslice(obj)
            size = (borders[1] - borders[0]) * (borders[3] - borders[2])
            borders_list.append(borders)
            sizelis.append(size)
        if get_largest:
            index = sizelis.index(max(sizelis))
            borders_list = [borders_list[index]]
        if len(borders_list) > 0:
            return borders_list[0]
        else:
            return None

    except Exception as e:
        raise RuntimeError("unable to generate borders from binary mask") from e


def rmblack(inp: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
    """
    Removes the black borders using strip based thresholding.
    Uses threshold as 0.2
    #TODO add threshold as argument
    Args:
        inp: numpy array or a dict

    Returns:
        array/ dict after removing the black borders

    """
    im = type_check(inp)
    # RM Black fails if image mean is 0 or standard deviation is close to 0
    # Removing those 2 conditions
    im_mean = np.mean(im)
    im_std = np.std(im)
    if im_mean > 0 and im_std / im_mean > 1:
        imarr = th_lower(im, 10)
        bds = get_borders(imarr, 0.2)  # default threshold
        imarr = im[bds[0] : bds[1], bds[2] : bds[3]]
    else:
        imarr = im
    return _good_return(inp, scale(imarr))


def rmblack_borders(inp: Union[np.ndarray, dict]) -> Tuple[int, int, int, int]:
    """
    Returns borders after running rmblack
    Args:
        inp: numpy array or a dict

    Returns:
        left, right, up, down borders (output of get_borders)

    """
    im = type_check(inp)
    imarr = th_lower(im, 10)
    bds = get_borders(imarr, 0.2)
    return bds


def compose(transforms: List[Callable]) -> Callable:
    """
    Takes in a list of tansform functions and returns a function that applies all the transforms on to an image
    Args:
        transforms: list of functions

    Returns:
        composed transform function

    """

    def composed_transform(im):
        for t in transforms:
            im = t(im)
        return im

    return composed_transform


def th_lower(inp: Union[np.ndarray, dict], p: float = -1) -> Union[np.ndarray, dict]:
    """
    # To be removed
    Clips the bottom p percentile of pixels out of the image
    clips the highest pixel value to 99th percentile pixel value
    Args:
        inp: numpy array or a dict
        p: percentile

    Returns:
        clipped image
    """
    arr = type_check(inp)
    if p == -1:
        p = randint(0, 20)
    ll = np.percentile(arr, p)
    narr = arr - ll
    narr = (narr > 0) * narr
    narr = scale(narr)
    ul = np.percentile(narr, 99)
    img = (narr > ul) * ul
    iml = (narr < ul) * narr
    farr = scale(img + iml)
    return _good_return(inp, farr)


def add_noise(inp: Union[np.ndarray, dict], p: float = -1) -> Union[np.ndarray, dict]:
    """
    Add gaussian noise to the image
    Args:
        inp: numpy array or a dict
        p: argument to the normal distribution used to generate noise

    Returns:
        noisy image
    """
    arr = type_check(inp)
    if p == -1:
        p = 0.01 * randint(2, 8)
    noise = np.random.normal(0, p, arr.shape)
    new_arr = arr + noise
    return _good_return(inp, scale(new_arr))


def smooth_noise(inp: Union[np.ndarray, dict], p: float = -1) -> Union[np.ndarray, dict]:
    """
    Add smoothed gaussian noise to the image
    Args:
        inp: numpy array or a dict
        p: argument to the normal distribution used to generate noise

    Returns:
        noisy image

    """
    arr = type_check(inp)
    if p == -1:
        p = 0.01 * randint(2, 8)
    noise = np.random.normal(0, p, arr.shape)
    noise = filters.gaussian(noise, 0.6)
    new_arr = arr + noise
    return _good_return(inp, scale(new_arr))


def smooth(inp: Union[np.ndarray, dict], p: float = -1) -> Union[np.ndarray, dict]:
    """
    Apply gaussian filter on the image to smooth it
    Args:
        inp: numpy array or a dict
        p: Strength of the gaussian filter to be used

    Returns:
        smoothed out image
    """
    arr = type_check(inp)
    if p == -1:
        p = randint(5, 12) * 0.1
    new_arr = filters.gaussian(arr, p)
    return _good_return(inp, scale(new_arr))


def mean_clip(inp: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
    """
    thresholds the array by it's mean and returns a scaled (0,1) array
    Args:
        inp: numpy array or a dict

    Returns:
        Array thresholded by it's mean and scaled
    """
    arr = type_check(inp)
    mn = arr.mean()
    new_arr = (arr > mn) * 1
    return _good_return(inp, scale(new_arr))


def lnorm(inp: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
    """
    Local normalization, normalize the input array by its
    mean and standard deviation
    Args:
        inp: np.ndarray or dict

    Returns:
        normalized image

    """
    arr = type_check(inp)
    arr = arr - arr.mean()
    arr = arr / (arr.std() + 0.000001)
    return _good_return(inp, arr)


def gaborr(inp: Union[np.ndarray, dict], fq: float = -1) -> Union[np.ndarray, dict]:
    """
    Apply gabor filter on the image
    Args:
        inp: np.ndarray or dict
        fq: frequency of gabor filter to be used
    Returns:
        output image from the gabor filter

    """
    arr = type_check(inp)
    if fq == -1:
        fq = 0.1 * randint(4, 7)
    narr, narr_i = filters.gabor(arr, frequency=fq)
    return _good_return(inp, scale(narr))


def resize_hard(size: int = 224) -> Callable:
    """
    resizes to a size and scales it #TODO is this redundant ?
    Args:
        size: size to which input should be resized to

    Returns:
        a function which resizes to size and scales the array
    """

    def rz(inp):
        im = type_check(inp)
        im = scale(im)
        # new_im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        new_im = resize(size, size)(im)
        return _good_return(inp, scale(new_im))

    return rz


def resize(height: int = 224, width: int = 224) -> Callable:
    """[summary]

    Args:
        height (int, optional): Image height to be resized to. Defaults to 224.
        width (int, optional): Image width to be resized to. Defaults to 224.

    Returns:
        Callable: function to do resizing
    """

    def rz(inp):
        im = type_check(inp)
        new_im = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
        return _good_return(inp, new_im)

    return rz


def resize_hard_3x(size: int = 224) -> Callable:
    """
    resizes array to 3s x s , s is input size,
    this is used for generating side images which are used by 3 stream models
    Args:
        size: size s

    Returns:
        a function to resize

    """

    def rz(inp):
        im = type_check(inp)
        im = scale(im)
        new_im = resize(3 * size, size)(im)
        # new_im = cv2.resize(im, (size, 3 * size), interpolation=cv2.INTER_AREA)
        return _good_return(inp, scale(new_im))

    return rz


def sub_smooth(inp: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
    """
    Subtract smoothed image from the image
    sharpens the edges in the image
    Args:
        inp: np.ndarray or dict

    Returns:
        image subtracted by its smoothed copy

    """
    im = type_check(inp)
    im = scale(im)
    nim = filters.gaussian(im, 8)
    nim = im - nim
    nim = scale(nim)
    return _good_return(inp, nim)


def scale_unity(inp: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
    """
    Scale the image between -1 and 1
    Args:
        inp: np.ndarray or dict

    Returns:
        scaled image

    """
    im = type_check(inp)
    im = scale(im)
    im = (im - 0.5) / 0.5
    return _good_return(inp, im)


def smooth_norm(inp: Union[np.ndarray, dict]) -> Union[np.ndarray, dict]:
    """
    Helps extract smooth and sharp features in an image
    iteratively add and subtract smoothed copies of image
    Args:
        inp: np.ndarray or dict

    Returns:
        transformed image

    """
    im = type_check(inp)
    blur1 = filters.gaussian(im, 7)
    blur2 = filters.gaussian(blur1, 7)
    blur3 = filters.gaussian(blur2, 7)
    im = im - blur1 + blur2 - blur3
    return _good_return(inp, scale(im))


def get_topi(diamask: np.ndarray) -> int:
    """
    Returns index of top most row with atleast one non-zero pixel in it, used on diaphragm segmenter output
    Args:
        diamask: numpy array

    Returns: index of top most row with atleast one non-zero pixel, defaults to zero

    """
    diamask = diamask.astype(int)
    h = diamask.shape[0]
    for i in range(h):
        strip = diamask[i, :]
        if strip.max() > 0:
            return i
    return 0


def get_bottomi(mask: np.ndarray) -> int:
    """
    Returns index of bottom most row with atleast one non-zero pixel in it, used on diaphragm segmenter output
    Args:
        mask: numpy array

    Returns: index of bottom most row with atleast one non-zero pixel, defaults to zero

    """
    mask = mask.astype(int)
    h = mask.shape[0]
    for i in range(h - 1, 0, -1):
        strip = mask[i, :]
        if strip.max() > 0:
            return i
    return 0


def centroid(m: Dict[str, float]) -> Tuple[float, float]:
    """
    returns centroid/ center of mass of a connected component
    Args:
        m: output of cv2.moments

    Returns:
        Center of mass of the connected component

    """
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return cx, cy


def mean_with_nonzero_std(arr: np.ndarray) -> np.ndarray:
    """
    Returns mean of array only for the dimensions where std/mean is non-zero.
    This is used after extracting pixel array from dicom when number of dimensions of
    the pixel array is > 2
    Args:
        arr: dicom pixel array

    Returns:
        mean array

    """
    assert arr.ndim > 2
    std = np.std(arr, axis=(0, 1))
    mean = np.mean(arr, axis=(0, 1))
    div = std / mean
    whr = np.where(div > 0.01)[0]

    if len(whr) == 0:
        mean_arr = arr[..., 0]
    elif len(whr) == 1:
        mean_arr = arr[..., whr[0]]
    else:
        mean_arr = np.mean(arr[..., tuple(whr)], axis=2)

    return mean_arr


def largest_connected_component(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Returns largest connected component of an array. Although this function works with boolean arrays,
    intended use is to be with the segmentation output and the corresponding threshold
    Args:
        mask: numpy array (segmentation output)
        threshold: threshold at which mask should be thresholded at

    Returns:
        boolean array with the largest connected component
    """
    # works even if the input array is bool
    _mask = (mask > threshold) * 1
    unique_values = np.unique(_mask)
    if len(unique_values) <= 1:
        return _mask

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        cast_uint8(_mask.astype(int)), connectivity=4
    )
    label_largest = np.argsort(stats[:, -1])[-2]
    output = _mask == label_largest

    return output


def get_convex_hull(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """
    generates a convex hull on a binary image
    Args:
        mask: expects a boolean numpy array
        thickness: thickness of the convexhull

    Returns:
        numpy array with convex hull of thickness

    """

    contours, hierarchy = cv2.findContours(
        cast_uint8(mask.astype(int)),
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_TC89_KCOS,
    )
    # find convex hull
    convex_hull = [cv2.convexHull(i) for i in contours]

    # draw convex hull
    blank_mask = np.zeros(mask.shape, np.uint8)
    draw_hull = cv2.drawContours(blank_mask, convex_hull, -1, 255, thickness=thickness)

    return draw_hull


def get_skeleton(convex_hull: np.ndarray) -> List[Tuple[int, int]]:
    """
    return skeleton of a convex hull, to be used to generate a skeleton for endotracheal tubes masks
    skimage skeletonize gives branches, this function avoid branches by averaging the x coordinates
    of the convex hull
    Args:
        convex_hull: numpy array

    Returns:
        Skeleton of the convex hull
    """
    assert convex_hull.dtype == np.uint8
    # get non-zero points
    x_hull, y_hull = np.nonzero(convex_hull)

    # get all unique x coords
    x_unique = np.unique(x_hull)

    # all the points of the skeleton
    skeleton = [(p, int(np.mean(y_hull[x_hull == p]))) for p in sorted(x_unique)]

    # removing the first point and final 2 points to remove small branches
    if len(skeleton) > 8:
        return skeleton[1:-3]
    else:
        return skeleton


def end_points(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """
    returns the endpoints of a skeleton using hit or miss transform
    Args:
        skeleton (np.ndarray): Boolean skeleton mask

    Returns:
        List[Tuple[int, int]]: [description]
    """
    input_image = cast_uint8(skeleton.astype(int))
    kernel = np.array(([-1, -1, -1], [-1, 1, -1], [-1, -1, -1]), dtype="int")

    output_image = np.zeros_like(input_image)

    for i in np.arange(0, 3):
        for j in np.arange(0, 3):
            if not (i == 1 and j == 1):
                kernel[i, j] = 1
                out = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel)
                kernel[i, j] = -1
                output_image = output_image + out

    points = []
    for (x, y) in zip(*np.nonzero(output_image)):
        points.append((int(x), int(y)))

    return points


def transform_point(pt: Tuple[int, int], left: int, up: int) -> Tuple[int, int]:
    """
    transform a point for rmblack

    Args:
        pt (Tuple[int, int]): point
        left (int): left rmblack border
        up (int): upper rmblack border

    Returns:
        Tuple[int, int]: transformed point
    """
    return (pt[0] + up, pt[1] + left)
