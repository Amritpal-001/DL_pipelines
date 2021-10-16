from typing import List, MutableMapping, Optional, Tuple, Union

import numpy as np

import qxr_utils.image.transforms as tsfms
from qxr_utils.preprocess import model_transforms


def pre_thorax_model_transform(
    im: np.ndarray, zoom_coords: Optional[Union[np.ndarray, List]] = None, im_size: int = 224
) -> Tuple[np.ndarray, Tuple[int, int], Union[np.ndarray, List]]:
    """Performs premodel transformations before passing it into thorax and diaphragm segmenters

    Args:
        im (np.ndarray): 2D numpy nd array image
        zoom_coords (Optional[Union[np.ndarray, List]], optional): zoom coordinates as list of 4 numbers. Defaults to None.
        im_size (int, optional): Image size to be passed into the model. Defaults to 224.

    Returns:
        np.ndarray: 1x1xim_sizexim_size numpy array to be passed to model
    """
    # Saving Image copy before doing transforms
    if zoom_coords is None:
        zoom_coords = tsfms.rmblack_borders(im)
    im = im[zoom_coords[0] : zoom_coords[1], zoom_coords[2] : zoom_coords[3]]
    im_height, im_width = im.shape
    patch_transform = tsfms.compose([tsfms.resize_hard(im_size), tsfms.lnorm])
    model_input = patch_transform(im)
    # TODO discuss: removing model_input[np.newaxis] as newaxis is added in dataloader in cxr_inference. cxr_product repo might break.
    # model_input = model_input[np.newaxis].astype(np.float32)
    model_input = model_input.astype(np.float32)
    zoom_coords = np.array(zoom_coords, ndmin=1)
    return model_input, (im_height, im_width), zoom_coords


def get_bottom_pixel_index_of_thorax_mask(
    thorax_single_mask: np.ndarray, diaphragm_single_mask: np.ndarray, extra_length_from_diaphragm_apex: float = 1.1
) -> int:
    """Getting the bottom pixel index of thorax mask after subtracting the diaphragm

    Args:
        thorax_single_mask (np.ndarray): square shaped 2D numpy array
        diaphragm_single_mask (np.ndarray): square shaped 2D numpy array, same as above
        extra_length_from_diaphragm_apex (int, optional): how much extra from diaphragm top. Defaults to 1.1.

    Returns:
        int: bottom pixel index of thorax mask
    """
    if thorax_single_mask.shape != diaphragm_single_mask.shape:
        diaphragm_single_mask = tsfms.resize(*np.shape(thorax_single_mask))(diaphragm_single_mask)
    diaphragm_top_pixel = float(tsfms.get_topi(diaphragm_single_mask))
    extra_length_from_diaphragm_apex = float(extra_length_from_diaphragm_apex)
    thorax_mask_height = np.shape(thorax_single_mask)[0]
    if extra_length_from_diaphragm_apex * diaphragm_top_pixel < thorax_mask_height:
        diaphragm_top_pixel = extra_length_from_diaphragm_apex * diaphragm_top_pixel
    else:
        diaphragm_top_pixel = thorax_mask_height
    return int(diaphragm_top_pixel)


def subtract_diaphragm(
    thorax_mask_left: np.ndarray,
    thorax_mask_right: np.ndarray,
    diaphragm_mask_left: np.ndarray,
    diaphragm_mask_right: np.ndarray,
    extra_length_from_diaphragm_apex: float = 1.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Takes left and right thorax, diaphragm masks and gives out the best thorax mask

    Args:
        thorax_mask_left (np.ndarray): left thorax mask 2D numpy array
        thorax_mask_right (np.ndarray): right thorax mask 2D numpy array
        diaphragm_mask_left (np.ndarray): left diaphragm mask 2D numpy array
        diaphragm_mask_right (np.ndarray): right diaphragm mask 2D numpy array
        extra_length_from_diaphragm_apex (int, optional): extra length from diaphragm apex. Defaults to 1.1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: [description]
    """
    left_thorax_bottom_index = get_bottom_pixel_index_of_thorax_mask(
        thorax_mask_left, diaphragm_mask_left, extra_length_from_diaphragm_apex
    )
    right_thorax_bottom_index = get_bottom_pixel_index_of_thorax_mask(
        thorax_mask_right, diaphragm_mask_right, extra_length_from_diaphragm_apex
    )
    bottom_most_index = max(left_thorax_bottom_index, right_thorax_bottom_index)
    # Creating a deep copy to ensure information is not lost in return function
    new_thorax_mask_left = np.copy(thorax_mask_left)
    new_thorax_mask_right = np.copy(thorax_mask_right)
    new_thorax_mask_left[bottom_most_index:, :] = 0
    new_thorax_mask_right[bottom_most_index:, :] = 0
    return new_thorax_mask_left, new_thorax_mask_right


def refine_borders(mask: np.ndarray, borders: List) -> List:
    """Refine Borders from given set of borders and Image input

    Args:
        mask (np.ndarray): 2D numpy image array, preferable bin mask
        borders (List): existing borders in a list format 1x4 input.

    Raises:
        RuntimeError: Raises runtime error if it is unable to generate borders

    Returns:
        List: each element in the list is a 1x4 numpy array.
    """
    try:
        new_borders = []
        for border in borders:
            zero_mask = np.zeros(np.shape(mask))
            zero_mask[border[0] : border[1], border[2] : border[3]] = 1
            new_mask = mask * zero_mask
            new_borders.append(tsfms.get_borders_with_scipy(new_mask))
        return new_borders
    except Exception as e:
        raise RuntimeError("unable to refine borders") from e


def get_lobe_borders(mask: np.ndarray, split: List) -> List:
    """Get lobe borders using a predefined split of list

    Args:
        mask (np.ndarray): 2D numpy array input image, preferable a bin mask
        split (List): list of 3 tuples each tuple is of 2 elements.

    Raises:
        RuntimeError: if borders can't be generated raises a Runtime Error

    Returns:
        List: List of elements where each element is a 1x4 numpy array
    """
    try:
        borders = tsfms.get_borders_with_scipy(mask)
        if borders is not None:
            height = borders[1] - borders[0]
            lobe_borders = []
            for ind in split:
                lobe_border = [
                    int(borders[0] + ind[0] * height),
                    int(borders[0] + ind[1] * height),
                    int(borders[2]),
                    int(borders[3]),
                ]
                lobe_borders.append(lobe_border)
            new_borders = refine_borders(mask, lobe_borders)
            return new_borders
        else:
            raise RuntimeError("Unable to generate Lobe Borders, since given borders is Null")
    except Exception as e:
        raise RuntimeError("Unable to generate Lobe Borders") from e


def get_lungmask_area(
    thorax_mask: np.ndarray, diaphragm_mask: np.ndarray, borders: List, im_size: int = 320
) -> Tuple[float, np.ndarray]:
    """Get Lung Area proportion in the complete image, the actual lung mask minus diaphragm.

    Args:
        thorax_mask (np.ndarray): 2D bin mask of thorax
        diaphragm_mask (np.ndarray): 2D bin mask of diaphragm
        borders (List): List of 3 numpy arrays, each element is of shape 1x4
        im_size (int, optional): Image width of the final mask. Height is 3x width. Defaults to 320.

    Returns:
        Tuple[float, np.ndarray]: area proportion, actual final mask
    """
    top, bottom = borders[0][0], borders[2][1]
    left, right = min([borders[i][2] for i in [0, 1, 2]]), max([borders[i][3] for i in [0, 1, 2]])
    border_slice = (slice(top, bottom), slice(left, right))
    thorax_mask = thorax_mask[border_slice]
    diaphragm_mask = diaphragm_mask[border_slice]
    final_mask = ((thorax_mask - diaphragm_mask) > 0) * 1
    final_mask = tsfms.resize_hard_3x(im_size)(final_mask)
    final_mask = (final_mask > 0) * 1
    return np.count_nonzero(final_mask == 1) / (3 * im_size * im_size), final_mask


def rescale_border_to_slice(
    border: Union[List, np.ndarray],
    im_height: int,
    im_width: int,
    mask_size: int,
    zoom_coords: Optional[Union[List, np.ndarray]] = None,
) -> Tuple[slice, slice]:
    """Rescales a border to original dimension and reonvert it into slice format

    Args:
        border (Union[List, np.ndarray]): border can be list of 4 ints or 1x4 numpy array
        im_height (int): image height
        im_width (int): image width
        mask_size (int): mask image height/width
        zoom_coords (Optional[Union[List, np.ndarray]], optional): zoom coordinates. Defaults to None.

    Returns:
        Tuple[slice, slice]: tuple of slice border output
    """
    new_border = [
        int(border[0] * im_height / mask_size),
        int(border[1] * im_height / mask_size),
        int(border[2] * im_width / mask_size),
        int(border[3] * im_width / mask_size),
    ]
    if zoom_coords is not None:
        new_border[0] += zoom_coords[0]
        new_border[1] += zoom_coords[0]
        new_border[2] += zoom_coords[2]
        new_border[3] += zoom_coords[2]
    return slice(new_border[0], new_border[1]), slice(new_border[2], new_border[3])


def get_side_sls_from_patches_dict(patches_dict: dict, side: str) -> Tuple[slice, slice]:
    """get side slices from patches_dict

    Args:
        patches_dict (dict): output dict from get_patches_dict function
        side (str): side for which side slices are to be generated

    Returns:
        Tuple[slice, slice]: Tuple of 2 slices
    """
    top = patches_dict[side][0][0].start
    bot = patches_dict[side][2][0].stop
    left = min([patches_dict[side][i][1].start for i in [0, 1, 2]])
    right = max([patches_dict[side][i][1].stop for i in [0, 1, 2]])
    return (slice(top, bot), slice(left, right))


def get_sides_dict_from_patches_dict(patches_dict: dict, bottom: bool = False) -> dict:
    """Returns the side borders dictionary from patches dictionary

    Args:
        patches_dict (dict): output dict from get_patches_dict
        bottom (bool, optional): to be generated for bottom patch or not. Defaults to False.

    Returns:
        dict: output dict contains keys of left and right each with their respective borders
    """
    left, right = "left", "right"
    if bottom:
        left, right = "left2", "right2"
    sides_dict = {
        left: get_side_sls_from_patches_dict(patches_dict, left),
        right: get_side_sls_from_patches_dict(patches_dict, right),
    }
    return sides_dict


def get_patches_dict(
    thorax_mask: np.ndarray,
    diaphragm_mask: np.ndarray,
    im_height: int,
    im_width: int,
    zoom_coords: Optional[Union[List, np.ndarray]] = None,
    diaphragm_threshold: float = 0.6,
    thorax_threshold: float = 0.5,
) -> Tuple[MutableMapping, Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    Function returns a six patches dict from thoraxmask, diaphragm mask, image dimensions.

    Args:
        thorax_mask (np.ndarray): 2D thorax mask, typically 224x224 shape output from Neural Network
        diaphragm_mask (np.ndarray): 2D diaphragm mask, typically 224x224 shape output from Neural Network
        im_height (int): zoomed image height
        im_width (int): zoomed image width
        zoom_coords (Optional[Union[List, np.ndarray]], optional): zoom coordinates. Defaults to None.
        diaphragm_threshold (float, optional): threshold for diaphragm mask. Defaults to 0.6.
        thorax_threshold (float, optional): threshold for thorax mask. Defaults to 0.5.

    Returns:
        Tuple[dict, Tuple[float, float], Tuple[np.ndarray, np.ndarray]]: outputs a tuple of 3 things.
        first is a dictionary of patch coordinates,
        second tuple is image area proportion,
        third tuple is side lung masks scaled
    """
    # saving thorax masks in a non-destructive way, might be useful later in the function
    diaphragm_lmask = (np.copy(diaphragm_mask[1]) > diaphragm_threshold) * 1
    diaphragm_rmask = (np.copy(diaphragm_mask[2]) > diaphragm_threshold) * 1
    mask_size = np.shape(thorax_mask[1])[0]

    # Get top pixel index of each diaphragm
    diaphragm_lmask_top_pixel = tsfms.get_topi(diaphragm_lmask)
    diaphragm_rmask_top_pixel = tsfms.get_topi(diaphragm_rmask)

    # lmask1, rmask1 refers to normal division which doesn't include the bottom of lung
    # lmask2, rmask2 refers to bottom division which include bottom of lung for peffusion, bluntedcp and pneumoperitoneum

    lmask1, rmask1 = subtract_diaphragm(thorax_mask[1], thorax_mask[2], diaphragm_lmask, diaphragm_rmask, 1.1)
    lmask2, rmask2 = subtract_diaphragm(thorax_mask[1], thorax_mask[2], diaphragm_lmask, diaphragm_rmask, 1.25)

    lmask1 = (tsfms.scale(lmask1) > thorax_threshold) * 1
    rmask1 = (tsfms.scale(rmask1) > thorax_threshold) * 1
    lmask2 = (tsfms.scale(lmask2) > thorax_threshold) * 1
    rmask2 = (tsfms.scale(rmask2) > thorax_threshold) * 1

    # predefined lung splits. this is will be deprecated from v4.
    normal_split = [(0, 0.32), (0.30, 0.62), (0.60, 1)]
    bottom_split = [(0, 0.282), (0.264, 0.545), (0.60, 1)]

    left_borders1 = get_lobe_borders(lmask1, normal_split)
    right_borders1 = get_lobe_borders(rmask1, normal_split)

    left_borders2 = get_lobe_borders(lmask2, bottom_split)
    right_borders2 = get_lobe_borders(rmask2, bottom_split)

    # calculating each lung area, side mask
    left_area, lmask_3x = get_lungmask_area(lmask1, diaphragm_lmask, left_borders1)
    right_area, rmask_3x = get_lungmask_area(rmask1, diaphragm_rmask, right_borders1)

    # TODO Change output type to a fixed class using pydantic
    out: MutableMapping = {
        "left": {},
        "right": {},
        "left2": {},
        "right2": {},
        "left_diaphragm_topi": int(diaphragm_lmask_top_pixel * im_height / mask_size),
        "right_diaphragm_topi": int(diaphragm_rmask_top_pixel * im_height / mask_size),
    }
    for i in range(3):
        out["left"][i] = rescale_border_to_slice(left_borders1[i], im_height, im_width, mask_size, zoom_coords)
        out["right"][i] = rescale_border_to_slice(right_borders1[i], im_height, im_width, mask_size, zoom_coords)
        if i < 2:
            out["left2"][i] = out["left"][i]
            out["right2"][i] = out["right"][i]
        else:
            out["left2"][i] = rescale_border_to_slice(left_borders2[i], im_height, im_width, mask_size, zoom_coords)
            out["right2"][i] = rescale_border_to_slice(right_borders2[i], im_height, im_width, mask_size, zoom_coords)

    if zoom_coords is not None:
        out["left_diaphragm_topi"] += zoom_coords[0]
        out["right_diaphragm_topi"] += zoom_coords[0]

    return out, (left_area, right_area), (lmask_3x, rmask_3x)


def get_side_borders_from_patch_borders(patch_borders: List) -> List:
    """function to get side border indices from patch border indices

    Args:
        patch_borders (List): shape 3x4

    Returns:
        List: List of 2 np array
    """

    def single_side(borders):
        top = borders[0][0]
        bottom = borders[2][1]
        left = min([borders[i][2] for i in [0, 1, 2]])
        right = max([borders[i][3] for i in [0, 1, 2]])
        return np.array([top, bottom, left, right])

    return [np.array(single_side(patch_borders[:3])), np.array(single_side(patch_borders[3:]))]


def give_patch_side_borders(patches_dict: dict, zoom_coords, bottom=False) -> Tuple[List, List, List, List]:
    """Return patch and side borders from patches dict

    Args:
        patches_dict (dict): patches dictionary
        bottom (bool, optional): [description]. Defaults to False.

    Returns:
        Tuple[List, List, List, List]: tuple of 4 iteams (patch slices, indices, side slices, indices)
    """
    left, right = "left", "right"
    if bottom:
        left, right = "left2", "right2"
    sides_dict = get_sides_dict_from_patches_dict(patches_dict, bottom)
    patch_borders_slice_list = [patches_dict[side][lobe] for side in [left, right] for lobe in range(3)]
    side_borders_slice_list = [sides_dict[side] for side in [left, right]]
    patch_borders_indices_list = [tsfms.get_indices_from_2dslice(sl, zoom_coords) for sl in patch_borders_slice_list]
    side_borders_indices_list = get_side_borders_from_patch_borders(patch_borders_indices_list)
    return patch_borders_slice_list, patch_borders_indices_list, side_borders_slice_list, side_borders_indices_list


def give_side_patch_nparray_borders(
    patches_dict: dict, im_sizes: List, fs_image: np.ndarray, zoom_coords, bottom=False
) -> MutableMapping:
    """Generates side and patch numpy arrays and borders
    Args:
        patches_dict (dict): Patches dictionary
        im_sizes (List): list of image sizes can be something like [224,320]
        fs_image (np.ndarray): 2D numpy array of original image
        zoom_coords  (array): array of zoom coordinates
        bottom (bool, optional): To be generated for normal image or bottom image. Defaults to False.

    Returns:
        dict: [description]

    """
    fs_image = fs_image[zoom_coords[0] : zoom_coords[1], zoom_coords[2] : zoom_coords[3]]
    (
        patch_borders_slice_list,
        patch_borders_indices_list,
        side_borders_slice_list,
        side_borders_indices_list,
    ) = give_patch_side_borders(patches_dict, zoom_coords, bottom)
    patches_nparray = "patches_nparray_bottom" if bottom else "patches_nparray"
    patch_borders = "patch_borders_bottom" if bottom else "patch_borders"
    sides_nparray = "sides_nparray_bottom" if bottom else "sides_nparray"
    side_borders = "side_borders_bottom" if bottom else "side_borders"
    # TODO Change output type to a fixed class using pydantic
    out: MutableMapping = {
        patches_nparray: {},
        patch_borders: patch_borders_indices_list,
        sides_nparray: {},
        side_borders: side_borders_indices_list,
    }
    for im_size in im_sizes:
        patches_list = [
            (model_transforms.get_val_patch_model_trasform(im_size)(fs_image[sl]))[np.newaxis]
            for sl in patch_borders_slice_list
        ]
        sides_list = [
            (model_transforms.get_val_side_model_transform(im_size)(fs_image[sl]))[np.newaxis]
            for sl in side_borders_slice_list
        ]
        out[patches_nparray][im_size] = np.concatenate(patches_list, axis=0)
        out[sides_nparray][im_size] = np.concatenate(sides_list, axis=0)
    return out
