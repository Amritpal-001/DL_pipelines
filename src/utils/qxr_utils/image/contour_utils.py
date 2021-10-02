from typing import List, Tuple

import cv2
import numpy as np
from scipy import interpolate as si

from qxr_utils.image import transforms as tsfms


def generate_smooth_contour(contour: List[np.ndarray]) -> List[np.ndarray]:
    """Function is used to smoothen a contour using scipy's interpolation function
        Currently using cubic interpolation for smoothening function

    Args:
        contour (List[np.ndarray]): contour as a list of sub contours
        each sub-contour is of Shape [N,1,2] -  N is number of points, each point is of shape 1x2
        and type is int

    Returns:
        List[np.ndarray]: smoothened contour
    """
    try:
        smooth_contour = []
        for cont in contour:
            x = list(cont[:, 0, 0])
            y = list(cont[:, 0, 1])
            assert len(x) >= 3, "minimum 3 points are needed to form a closed figure"
            orig_len = len(x) + 1
            x = x[-2:] + x + x[:3]
            y = y[-2:] + y + y[:3]

            t = np.arange(len(x))
            ti = np.linspace(2, orig_len + 1, 10 * orig_len)

            xi = si.interp1d(t, x, kind="cubic")(ti)
            yi = si.interp1d(t, y, kind="cubic")(ti)

            z = np.column_stack((xi.astype(int), yi.astype(int)))
            z = z[:, np.newaxis, :]
            smooth_contour.append(z)
        return smooth_contour
    except Exception as e:
        # Returns normal contour if exception in finding contour from smooth contour
        print(e)
        return contour


def generate_smooth_solid_mask(mask: np.ndarray, area_th: float = 900, contour_type: str = "convex") -> np.ndarray:
    """[summary]

    Args:
        mask (np.ndarray): 2D numpy array for which smooth mask is to be generated
        area_th (float, optional): area threshold for which contours are to be ignored. Defaults to 900.
        contour_type (str, optional): convex or contour area type. Defaults to "convex".

    Returns:
        np.ndarray: smoothened smooth solid mask after the function
    """
    mask = tsfms.cast_uint8(mask * 1)
    contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_KCOS)
    # Removing contours which are less than threshold area
    cont_th = []
    for cont in contours:
        if cv2.contourArea(cont) > area_th:
            cont_th.append(cont)
    contours = cont_th
    if contour_type == "convex":
        contours = [cv2.convexHull(contour, True) for contour in contours]
    contours = generate_smooth_contour(contours)
    blank_mask = np.zeros(mask.shape, np.uint8)
    solid = cv2.drawContours(blank_mask, contours, -1, 255, cv2.FILLED)
    solid = (cv2.blur(solid, ksize=(41, 41)) > 127) * 1
    solid = (solid > 0) * 1
    return solid


def pad_blur_unpad(mask: np.ndarray, pad_percentage: int = 20) -> np.ndarray:
    """Function adds a pad to the mask, Blurs the mask and then unpads the mask

    Args:
        mask (np.ndarray): 2D numpy ndarray generated generally an image
        pad_percentage (int, optional): percentage extra area for which we should pad and blur. Defaults to 20.

    Returns:
        np.ndarray: returns mask of the same shape after performing the function
    """
    # Pad the mask by 20% as default
    extra_area = 1 + (pad_percentage / 100)
    mask_pad_shape = (
        int(np.shape(mask)[0] * extra_area),
        int(np.shape(mask)[1] * extra_area),
    )
    mask_pad = np.zeros(mask_pad_shape, dtype=np.uint8)

    mask_left = np.shape(mask_pad)[0] // 2 - np.shape(mask)[0] // 2
    mask_top = np.shape(mask_pad)[1] // 2 - np.shape(mask)[1] // 2
    mask_right = mask_left + np.shape(mask)[0]
    mask_bottom = mask_top + np.shape(mask)[1]

    mask_pad[mask_left:mask_right, mask_top:mask_bottom] = mask
    # applying gaussian blur of kernel size 81x81
    mask_pad = (cv2.blur(mask_pad, ksize=(81, 81)) > 127) * 1
    mask = mask_pad[mask_left:mask_right, mask_top:mask_bottom]
    mask = tsfms.cast_uint8(mask)
    return mask


def find_contour(
    mask: np.ndarray,
    solid_mask: bool = False,
    contour_type: str = "concave",
    color: tuple = (255, 255, 255),
    thickness: int = 10,
) -> Tuple[np.ndarray, float, List[np.ndarray]]:
    """Finds the contour of a solid mask, the final contour is a smoothened contour after blur, scipy interpolation
     functions
    being applied. Option to return a solid contour or a line contour is possible through arguments.

    Args:
        mask (np.ndarray): 2D np array which has masks.
        solid_mask (bool, optional): If we output as a solid inside the contour or not. Defaults to False.
        color (tuple, optional): final color of the contour. Defaults to (255, 255, 255).
        thickness (int, optional): contour line thickness. Defaults to 10.

    Returns:
        Tuple[np.ndarray, float, List[np.ndarray]]: gives a tuple of 2D image which has a contour, area of the contour,
         contour points
    """
    mask = tsfms.cast_uint8(mask)
    mask_orig = np.copy(mask)
    orig_mask_shape = np.shape(mask_orig)
    # Resizing image to 1280 so that, the transforms, kernel size doesn't effect a lot
    mask_shape = (1280, int(1280 * orig_mask_shape[1] / orig_mask_shape[0]))
    mask = cv2.resize(mask, (mask_shape[1], mask_shape[0]), cv2.INTER_AREA)
    contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_KCOS)
    # smoothing solid contour masks with scipt, blur
    contours = generate_smooth_contour(contours)
    blank_mask = np.zeros(mask.shape, np.uint8)
    solid = cv2.drawContours(blank_mask, contours, -1, 255, cv2.FILLED)
    solid = pad_blur_unpad(solid)
    solid = cv2.resize(solid, (orig_mask_shape[1], orig_mask_shape[0]), cv2.INTER_AREA)
    # Smoothing the actual solid mask
    contours, hierarchy = cv2.findContours(solid, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_KCOS)
    out_area = 0.0
    default_smooth = 0.05
    total_contours = len(contours)
    smooth_contours = []
    # Concave smoothing
    if contour_type != "concave":
        contours = [cv2.convexHull(contour, True) for contour in contours]
    for i in range(total_contours):
        cnt = contours[i]
        smooth = default_smooth
        if int(smooth * len(cnt)) == 0:
            smooth = 5 / len(cnt)
        smooth_contours.append(cnt[0 : len(cnt) : int(len(cnt) * smooth)].astype(int))
    contours = smooth_contours
    cnts = []
    # Sanity check of areas
    for cont in contours:
        if cv2.contourArea(cont) > 2000:
            out_area = out_area + cv2.contourArea(cont)
            cnts.append(cont)
    contours = cnts
    if solid_mask:
        blank_mask = np.zeros(mask_orig.shape, np.uint8)
        draw_contour = cv2.drawContours(blank_mask, contours, -1, 255, cv2.FILLED)
    else:
        blank_mask = np.zeros([*mask_orig.shape, 3], np.uint8)
        draw_contour = cv2.drawContours(blank_mask, contours, -1, color, thickness)
        draw_contour = cv2.cvtColor(draw_contour, cv2.COLOR_BGR2GRAY)
    return draw_contour, out_area, contours
