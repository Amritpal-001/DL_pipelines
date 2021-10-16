import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from qtranslate import Translator
from scipy.spatial.distance import cdist

from qxr_utils.image import contour_utils as cu
from qxr_utils.image import transforms as tf
from qxr_utils.postprocess.contour import constants as constants_contour
from qxr_utils.tag_config import constants as constants_tag_config
from qxr_utils.tag_config.constants import GeneralKeys as GK
from qxr_utils.tag_config.constants import contour_reverse_tags_ids, contour_tags_ids

# TODO make contour generator a class which takes preds dict, usecase model as input


def generate_contour(preds_dict: dict, language: str = "en") -> dict:
    # TODO add pydantic class for mask, which is the output of this function
    # TODO add use case config as input
    """Generates contours which are feedable to secondary capture repo

    Args:
        preds_dict (dict): Standard predictions dictionary with tags and its results
        language (str, optional): chosing which language. Defaults to "en". Can
        be one of en, es, fr, pt for english, spanish, french and portuguese respectively

    Returns:
        dict: Dictionary of masks with the following format outputs: mask, area, contours
    """
    mask: Dict[Any, Dict[str, Dict]] = {}
    # mask_tags = [ContourMaskTags.lung, ContourMaskTags.covid, ContourMaskTags.bones, ContourMaskTags.critical]
    for mask_tag in constants_tag_config.TAGS_TO_COMBINE:
        mask[mask_tag] = {
            "mask": {"left": None, "right": None},
            "area": {"left": 0, "right": 0},
            "contours": {"left": [], "right": 0},
            "count": {"left": 0, "right": 0},
        }
        for tag in getattr(constants_tag_config.ContoursToCombine, mask_tag).value:
            if tag in preds_dict and preds_dict[tag][GK.ensemble_pixel.value] is not None:
                lmask = tf.np_decompress(preds_dict[tag][GK.ensemble_pixel.value][GK.mask.value][0])
                rmask = tf.np_decompress(preds_dict[tag][GK.ensemble_pixel.value][GK.mask.value][1])
                left_area, right_area = preds_dict[tag][GK.ensemble_pixel.value]["area"]
                lmask = lmask.astype(int) * (2 ** contour_tags_ids[tag])
                rmask = rmask.astype(int) * (2 ** contour_tags_ids[tag])
                if mask[mask_tag]["mask"]["left"] is None:
                    mask[mask_tag]["mask"] = {"left": lmask, "right": rmask}
                    mask[mask_tag]["area"] = {"left": left_area, "right": right_area}
                else:
                    mask[mask_tag]["count"]["left"] += 1 * (np.sum(lmask) > 0)
                    mask[mask_tag]["count"]["right"] += 1 * (np.sum(rmask) > 0)
                    mask[mask_tag]["mask"]["left"] = mask[mask_tag]["mask"]["left"] + lmask
                    mask[mask_tag]["mask"]["right"] = mask[mask_tag]["mask"]["right"] + rmask
        if mask[mask_tag]["mask"]["left"] is not None:
            # generating contour for left lung
            default_contour_type = "concave"
            if mask[mask_tag]["count"]["left"] > 1:
                default_contour_type = "convex"
            left_contour, left_area, left_ctr_points = cu.find_contour(
                (mask[mask_tag]["mask"]["left"] > 0) * 1, solid_mask=False, contour_type=default_contour_type
            )
            # generating contour for right lung
            default_contour_type = "concave"
            if mask[mask_tag]["count"]["right"] > 1:
                default_contour_type = "convex"
            right_contour, right_area, right_ctr_points = cu.find_contour(
                (mask[mask_tag]["mask"]["right"] > 0) * 1, solid_mask=False, contour_type=default_contour_type
            )
            # generating individual contours
            left_contours = give_individual_contours(
                mask[mask_tag]["mask"]["left"], left_ctr_points, mask[mask_tag]["area"]["left"], language
            )
            right_contours = give_individual_contours(
                mask[mask_tag]["mask"]["right"], left_ctr_points, mask[mask_tag]["area"]["right"], language
            )
            mask[mask_tag]["contours"]["left"] = left_contours
            mask[mask_tag]["contours"]["right"] = right_contours
            mask[mask_tag]["mask"]["left"] = tf.np_compress(mask[mask_tag]["mask"]["left"])
            mask[mask_tag]["mask"]["right"] = tf.np_compress(mask[mask_tag]["mask"]["right"])
    return mask


def give_individual_contours(mask: np.ndarray, ctr_points: list, lung_area: int, language: str = "en") -> list:
    # TODO add pydantic class for output of this function
    """Function to give individual contour label points from a mask along with its label

    Args:
        mask (np.ndarray): Mask output intermediate in generate_contours function
        ctr_points (list): Individual contour points for this mask
        lung_area (int): Area of this mask
        language (str, optional): Language for our contours. Defaults to "en".

    Returns:
        list: list of all contours, each contour is a dict with points, tag names, label and area
    """
    all_contours = []
    for ctr in ctr_points:
        blank_mask = np.zeros(mask.shape, np.uint8)
        solid_contour = (cv2.drawContours(blank_mask, [ctr], -1, 255, cv2.FILLED) > 0) * 1
        area = cv2.contourArea(ctr)
        contour_tags = give_contour_tags(mask, solid_contour)
        ctr_dict = {
            "points": ctr,
            "tags": contour_tags,
            "label": refine_contour_label(contour_tags, language),
            "area": round(100 * (area / lung_area)),
        }
        all_contours.append(ctr_dict)
    return all_contours


def give_contour_tags(mask: np.ndarray, solid_contour: np.ndarray) -> list:
    """Gives out the exact contours present in a particular blob

    Args:
        mask (np.ndarray): Integer multiplied mask from stage 1,
                        where we multiply each contour with its label
        solid_contour (np.ndarray): solid mask

    Returns:
        list: list of tags present for this contour
    """
    mask = np.multiply(mask, solid_contour)
    tag_bits_int = np.bitwise_or.reduce(np.bitwise_or.reduce(mask))
    tag_bits = "{0:b}".format(tag_bits_int)
    contour_tags = []
    for tag_bit in range(len(tag_bits)):
        pos = len(tag_bits) - tag_bit - 1
        if tag_bits[pos] == "1":
            contour_tags.append(contour_reverse_tags_ids[tag_bit])
    return contour_tags


def give_opacity_tags_vector(list_of_tags: list) -> Tuple[list, str]:
    """outputs opacity tags vector from a list of output tags

    Args:
        list_of_tags (list): List of tag outputs from preds dict

    Returns:
        Tuple[list, str]: final list of tags after removing duplicates and the opacity vector
    """
    # value of this dict is a list whose values correspond to the index in the
    # opacity feature vector
    _dict = {
        GK.opacity.value: [0],
        GK.nodule.value: [0, 1],
        GK.consolidation.value: [0, 2],
        GK.fibrosis.value: [0, 3],
        GK.atelectasis.value: [0, 4],
        GK.calcification.value: [0, 4],
        GK.reticulonodularpattern.value: [0, 4],
    }
    out = [0, 0, 0, 0, 0]

    for tag in _dict:
        if tag in list_of_tags:
            for i in _dict[tag]:
                out[i] = 1
            list_of_tags.remove(tag)

    return list_of_tags, "".join(map(str, out))


def refine_contour_label(labels: list, language: str = "en") -> str:
    """Refine the contour label by removing dependent tags

    Args:
        labels (list): list of tags present in a contour
        language (str, optional): language output for the label. Defaults to "en".

    Returns:
        str: string format of the output label
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    translator_path = os.path.join(cur_dir, "../../tag_config/translations")
    translator = Translator(translator_path)
    # translator = Translator(str(pathlib.Path(__file__).parent.parent.parent / "tag_config" / "translations"))
    translator.load(language, "contour")
    if GK.bluntedcp.value in labels and GK.pleuraleffusion.value in labels:
        labels.remove(GK.bluntedcp.value)
    labels, opacity_tags_vector = give_opacity_tags_vector(labels)
    TAG_PREFIX = "tag_"
    labels = [translator.get(language, TAG_PREFIX + i) for i in labels]
    if len(labels) != 0:
        if labels[0] is None:
            labels = []

    if opacity_tags_vector != "00000":
        opacity_keywords = constants_contour.opacity_labels[opacity_tags_vector]
        opacity_translations = [translator.get(language, i) for i in opacity_keywords]
        opacity_string = opacity_translations[0] + "\n(" + ", ".join(opacity_translations[1:]) + ")"
        labels.append(opacity_string)

    return ",\n".join(labels)


def get_contours_update_preds_dict(preds_dict: dict) -> Tuple[Dict[str, Dict], Dict]:
    """update the contours to predictions dictionary

    Args:
        preds_dict (dict): standard predictions dictionary output from cxr product repo

    Returns:
        [type]: the same predictions dict with additional keys
    """
    contour_keys = list(preds_dict["contours"])
    contours: Dict[str, Dict] = {}
    contour_counter = 0

    for side in ["left", "right"]:
        contours[side] = {}
        for tag in contour_keys:
            contour_list = preds_dict["contours"][tag]["contours"][side]
            for ct in contour_list:
                ct["contour_label"] = contour_counter
                contours[side][contour_counter] = ct["points"]
                contour_counter += 1

    return contours, preds_dict


def get_horizontal_start(input_dict: dict) -> dict:
    """function returns the horizontal start point

    Args:
        input_dict (dict): input dictionary which we get from pre tag prediction

    Returns:
        dict: left and right coordinates in a dictionary
    """
    lsb = input_dict["side_borders"][0]
    rsb = input_dict["side_borders"][1]
    start = {"left": lsb[-1], "right": rsb[-2]}
    return start


def get_side_box_points(
    contours: list,
    shape: Tuple[int, int],
    top_padding: int = 100,
    bottom_padding: int = 100,
    horizontal_offset: int = 100,
    horizontal_start: Optional[int] = None,
) -> dict:
    """outputs side box points for each contour label

    Args:
        contours (list): list of all the contours
        shape (Tuple[int, int]): image shape
        top_padding (int, optional): Padding from the top. Defaults to 100.
        bottom_padding (int, optional): Padding from the bottom. Defaults to 100.
        horizontal_offset (int, optional): total horizontal offset. Defaults to 100.
        horizontal_start (int, optional): start horizontal point. Defaults to None.

    Returns:
        dict: output for each contour label
    """
    assert type(contours), list
    assert len(contours) > 0, "zero contours in this input"

    moments = [cv2.moments(c[1]) for c in contours]
    centroids = {tf.centroid(m): c for m, c in zip(moments, contours)}
    height, width = shape

    if horizontal_start is None:
        horizontal_start - width

    centroids_count = len(centroids)
    sorted_centroids = dict(sorted(centroids.items(), key=lambda x: x[0][1]))

    y_boxes = np.linspace(top_padding, height - bottom_padding, centroids_count + 1, False, dtype=int)[1:]
    x_boxes = [horizontal_start + horizontal_offset] * len(y_boxes)
    points_of_boxes = map(tuple, zip(x_boxes, y_boxes))

    out = {}
    for item, pt in zip(sorted_centroids.items(), points_of_boxes):
        contour_label = item[1][0]  # item -> centroid, (label, pts)
        start = item[0]
        contour_pts = np.squeeze(item[1][1])
        end = pt

        # find the point on the contour which is between centroid and box point
        distance1 = cdist(np.expand_dims(start, axis=0), contour_pts)
        distance2 = cdist(np.expand_dims(end, axis=0), contour_pts)
        ix = np.argmin(distance1 + distance2)
        mid = tuple(contour_pts[ix])

        out[contour_label] = [mid, end]

    return out


def update_all_box_points_to_preds_dict(input_dict: dict, preds_dict: dict) -> dict:
    """Function updates all the box points to the predictions dictionary

    Args:
        input_dict (dict): input dictionary after pretag prediction
        preds_dict (dict): output dictionary after predictions are computed

    Returns:
        dict: predictions dictionary after updating the keys with contours
    """
    shape = input_dict["fsnparray"].shape
    start = get_horizontal_start(input_dict)
    contours_dict, preds_dict = get_contours_update_preds_dict(preds_dict)
    contours_keys = list(preds_dict["contours"])

    offset = {"left": 200, "right": -200}

    lsb, rsb = input_dict["side_borders"][:2]
    paddings = {"left": (lsb[0], shape[0] - lsb[1]), "right": (rsb[0], shape[0] - rsb[1])}

    box_points = {}
    for side in ["left", "right"]:
        side_contours = contours_dict[side]
        label_contours = list(side_contours.items())
        if len(side_contours) > 0:
            top_padding = paddings[side][0]
            bottom_padding = paddings[side][1]
            box_points_side = get_side_box_points(
                label_contours,
                shape=shape,
                top_padding=top_padding,
                bottom_padding=bottom_padding,
                horizontal_start=start[side],
                horizontal_offset=offset[side],
            )
            box_points.update(box_points_side)

            # adding box points to label
            for tag in contours_keys:
                contour_list = preds_dict["contours"][tag]["contours"][side]
                for ct in contour_list:
                    label = ct["contour_label"]
                    if label in box_points_side:
                        ct["box_points"] = box_points_side[label]

    return preds_dict
