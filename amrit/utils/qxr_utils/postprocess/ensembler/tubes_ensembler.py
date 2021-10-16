import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from pydantic import validator

from qxr_utils.image import transforms as tf
from qxr_utils.postprocess.contour import tubes_contour as tc
from qxr_utils.postprocess.ensembler import base
from qxr_utils.tag_config.classes import UseCaseModel
from qxr_utils.tag_config.constants import GeneralKeys as GK

logger = logging.getLogger("TUBES ENSEMBLER")


def tube_contour(
    height: int, width: int, tube: tc.Array, tube_type: str, min_pix: int = 10, offset: int = 0
) -> List[Optional[List[tc.POINT]]]:
    """
    Returns tube contour points scaled to original image dimensions
    Args:
                    height: original image height
                    width: original image width
                    tube: boolean mask of tube from segmentation output after thresholding
                    tube_type: breathingtube or gastrictube
                    min_pix: minimum number of pixels per branch. If a branch length is less than this, it will be pruned off
                    offset: to translate the points so that the contour won't intersect with the actual tube

    Returns:
                    list of list of points, each list corresponds to one segment

    """
    assert tube.shape == (512, 512), "tube shape is not 512,512"

    resized_offset = int(offset * width / 512)
    data = {"mask": tube, "min_pix": min_pix}
    _cnt = tc.TubeContour(**data)
    points = []
    try:
        points = _cnt.draw_contours(tube_type=tube_type, shape=(height, width), offset=resized_offset)
    except Exception as e:
        logging.exception(e)

    return points


class BreathingTubesEnsembler(base.BaseEnsembler):
    use_case_model: UseCaseModel
    tag_name = GK.breathingtube.value
    preds_dict: Dict[str, dict]
    preprocessing_dict: Dict[str, Any]

    @validator("preprocessing_dict")
    def check_preprocessing_dict(cls, v):
        """
        checks if preprocessing_dict has necessary values
        Args:
                        v: preprocessing_dict

        Returns:
                        preprocessing_dict after validation

        """
        required_keys = [
            GK.original_width,
            GK.original_height,
            GK.rmblk_borders,
            GK.pixel_spacing,
        ]

        for i in required_keys:
            assert i.value in v, f"{i.value} not available in preprocessing_dict"

        return v

    def draw_carina(self, height: int, width: int, carina: tc.Array) -> Dict:
        """
        return carina points scaled to the original image dimensions
        Args:
                        height: original image height
                        width: original image width
                        carina: carina segmentation mask after thresholding

        Returns:
                        a dictionary with carina centroid, boolean mask of the arrow and list of points for the arrow

        """
        assert carina.shape == (512, 512), "carina shape is not 512,512"
        carina_arrow = np.zeros((height, width), dtype=np.uint8)
        carina_centroid = None
        carina_pts = []

        try:
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
                tf.cast_uint8(carina.astype(int)), connectivity=4
            )
            carina_label = np.argsort(stats[:, -1])[-2]
            carina_centroid = tuple(centroids[carina_label].astype(int))
            x, y, w, h, area = stats[carina_label]

            h_old, w_old = carina.shape
            h_new, w_new = carina_arrow.shape
            cx, cy = carina_centroid

            cx = int(cx * w_new / w_old)
            cy = int(cy * h_new / h_old)

            x1 = cx - int(5 * h_new / 512)
            x2 = cx + int(5 * h_new / 512)
            y1 = cy + int(5 * h_new / 512)

            cv2.line(carina_arrow, (x1, y1), (cx, cy), (255, 255, 255), thickness=5)
            cv2.line(carina_arrow, (cx, cy), (x2, y1), (255, 255, 255), thickness=5)

            carina_pts = [(x1, y1), (cx, cy), (x2, y1)]

        # TODO need better exception handling
        except Exception as e:
            logging.exception(e)

        result = {"carina_centroid": carina_centroid, "carina_arrow": carina_arrow, "carina_pts": carina_pts}

        return result

    def ensemble(self) -> base.EnsembledResults:
        """

        Returns: ensembled results for breathing tube

        """
        # allowed distance between tip of the tube and carina centroid in cm,
        # to 3 cm
        ALLOWED_DISTANCE_IN_CM = self.tag_info.extras.get(GK.allowed_distance_in_cm.value, 3)
        # allowed distance between tip of the tube and carina centroid in
        # pixels, defaults to 35 pixels
        ALLOWED_DISTANCE_IN_PIXELS = self.tag_info.extras.get(GK.allowed_distance_in_pixels.value, 35)

        original_height = self.preprocessing_dict[GK.original_height.value]
        original_width = self.preprocessing_dict[GK.original_width.value]
        rmblk_l, _, rmblk_u, _ = self.preprocessing_dict[GK.rmblk_borders.value]
        pixel_spacing = self.preprocessing_dict[GK.pixel_spacing.value]
        softmax_probs = {}

        for _tag in self.tag_info.dependent_abnormality_prediction_models:
            softmax_probs[_tag] = self.preds_dict[_tag]
        ett_scan_ensembled = softmax_probs[GK.endotracheal_tube_cls.value][GK.ensemble_scan.value]
        tt_scan_ensembled = softmax_probs[GK.tracheostomy_tube_cls.value][GK.ensemble_scan.value]

        # find if it is endotracheal or tracheostomy tube
        if ett_scan_ensembled >= tt_scan_ensembled:
            scan_tag = GK.endotracheal_tube_cls.value
            pixel_tag = GK.endotracheal_tube_seg.value
            scan_prob = ett_scan_ensembled
        else:
            scan_tag = GK.tracheostomy_tube_cls.value
            pixel_tag = GK.tracheostomy_tube_seg.value
            scan_prob = tt_scan_ensembled

        # threshold scan score
        prediction_model_info = self.prediction_models_info[scan_tag]
        tube_presence = scan_prob > prediction_model_info.scan_threshold

        pixel_probs_ensembled = softmax_probs[pixel_tag][GK.ensemble_pixel.value]

        # threshold pixel probabilities
        pixel_predictions = pixel_probs_ensembled > prediction_model_info.pixel_threshold
        # multiply pixel_predictions with it's presence
        pixel_predictions = tube_presence * pixel_predictions
        # change tube presence if pixel_predictions has no +ve pixels
        tube_presence = tube_presence and np.sum(pixel_predictions) > 0

        tube_position = None
        # breathing_tube = np.zeros((original_height, original_width), dtype=bool)
        prediction = 0
        tube_tip = None
        carina_y = None
        transformed_points = []
        vertical_distance_in_cm = -1

        # find if the tube is in position
        if tube_presence:
            # tube tip
            try:
                (
                    nb_components,
                    output,
                    stats,
                    centroids,
                ) = cv2.connectedComponentsWithStats(tf.cast_uint8(pixel_predictions.astype(int)), connectivity=4)

                # largest connected component to remove artifacts
                tube_label = np.argsort(stats[:, -1])[-2]
                x, y, w, h, area = stats[tube_label]
                tube_pixel = output == tube_label
                tube_tip = y + h

                # carina predictions
                carina_probs = softmax_probs[GK.carina.value][GK.pixel_scores.value]
                # ensemble
                carina_probs_ensembled = np.mean(carina_probs, axis=0)
                carina_info = self.prediction_models_info[GK.carina.value]
                # threshold
                carina_threshold = carina_info.pixel_threshold
                carina_preds = carina_probs_ensembled > carina_threshold

                carina_out = self.draw_carina(original_height, original_width, carina_preds)

                carina_centroid = carina_out["carina_centroid"]
                carina_arrow = carina_out["carina_arrow"]
                carina_pts = carina_out["carina_pts"]

                # check if carina not null
                if carina_centroid is None or carina_arrow is None or carina_centroid is None:
                    tube_position = -1  # position indeterminate
                else:
                    carina_y = carina_centroid[1]
                    vertical_distance = carina_y - tube_tip
                    print(f"vertical distance is {vertical_distance} px")
                    if pixel_spacing != -1:
                        # pixel spacing * pixels will give distance in mm, diving by 10 to get it in cm
                        vertical_distance *= original_height / (512 * 10)
                        print(f"vertical distance is {vertical_distance} cm")
                        vertical_distance_in_cm = vertical_distance
                        if vertical_distance >= ALLOWED_DISTANCE_IN_CM:
                            tube_position = 0
                        else:
                            tube_position = 1
                            prediction = 1

                    else:
                        if vertical_distance >= ALLOWED_DISTANCE_IN_PIXELS:
                            tube_position = 0
                        else:
                            tube_position = 1

                # get contour related items from tag info
                min_pix = self.tag_info.extras.get(GK.min_pix.value, 10)
                offset = self.tag_info.extras.get(GK.offset.value, 10)

                tube_contour_points = tube_contour(
                    original_height,
                    original_width,
                    tube_pixel,
                    min_pix=min_pix,
                    offset=offset,
                    tube_type=self.tag_name,
                )

                transformed_points = [
                    [tf.transform_point(pt, rmblk_l, rmblk_u)] for pt_list in tube_contour_points for pt in pt_list
                ]

                # transformed_points = [tf.transform_point(i, rmblk_l, rmblk_u) for i in tube_contour_points[0]]

                if carina_pts is not None:
                    transformed_points.append(carina_pts)

            except Exception as e:
                logging.exception(e)

        _extras = {
            GK.tube_presence.value: tube_presence,  # 0 if absent , 1 if present
            # None if absent, 0 if in position, 1 if out of position, -1 if indeterminate
            GK.tube_position.value: tube_position,
            GK.tube_tip.value: int(tube_tip) if tube_tip is not None else None,
            GK.carina.value: int(carina_y) if carina_y is not None else None,
            GK.points.value: transformed_points,  # list of list of points
            GK.vertical_distance_in_cm.value: vertical_distance_in_cm,
        }

        results_dict = {
            GK.score.value: scan_prob,
            GK.prediction.value: prediction,  # prediction is 1 only if tube is out of position
            GK.extras.value: _extras,
            # GK.ensemble_pixel.value: breathing_tube,
        }

        ensembled_results = base.EnsembledResults(**results_dict)

        return ensembled_results


class GastricTubeEnsembler(base.BaseEnsembler):
    use_case_model: UseCaseModel
    tag_name = GK.gastrictube.value
    preds_dict: Dict[str, dict]
    preprocessing_dict: Dict[str, Any]

    @validator("preprocessing_dict")
    def check_preprocessing_dict(cls, v):
        """
        checks if preprocessing_dict has necessary values
        Args:
                        v: preprocessing_dict

        Returns:
                        preprocessing_dict after validation

        """
        required_keys = [
            GK.original_width,
            GK.original_height,
            GK.rmblk_borders,
            GK.left_diaphragm_topi,
            GK.right_diaphragm_topi,
        ]

        for i in required_keys:
            assert i.value in v, f"{i.value} not available in preprocessing_dict"

        return v

    def ensemble(self) -> base.EnsembledResults:
        """

        Returns: ensembled results for gastric tube

        """
        original_height = self.preprocessing_dict[GK.original_height.value]
        original_width = self.preprocessing_dict[GK.original_width.value]
        rmblk_l, _, rmblk_u, _ = self.preprocessing_dict[GK.rmblk_borders.value]
        left_diaphragm_topi = self.preprocessing_dict[GK.left_diaphragm_topi.value]
        right_diaphragm_topi = self.preprocessing_dict[GK.right_diaphragm_topi.value]

        softmax_probs = {}

        scan_tag = GK.nasogastric_tube_cls.value
        pixel_tag = GK.nasogastric_tube_seg.value

        for _tag in self.tag_info.dependent_abnormality_prediction_models:
            softmax_probs[_tag] = self.preds_dict[_tag]
        scan_prob_ensembled = softmax_probs[scan_tag][GK.ensemble_scan.value]

        # threshold scan score
        prediction_model_info = self.prediction_models_info[scan_tag]
        tube_presence = scan_prob_ensembled > prediction_model_info.scan_threshold

        # ensemble pixel probabilites
        pixel_probs_ensembled = softmax_probs[pixel_tag][GK.ensemble_pixel.value]
        # threshold pixel probabilities
        pixel_predictions = pixel_probs_ensembled > prediction_model_info.pixel_threshold
        # multiply pixel_predictions with it's presence
        pixel_predictions = tube_presence * pixel_predictions
        # change tube presence if pixel_predictions has no +ve pixels
        tube_presence = tube_presence and np.sum(pixel_predictions) > 0

        tube_position = None
        # gastric_tube = np.zeros((original_height, original_width), dtype=bool)
        prediction = 0
        tube_tip = None
        diaphragm_topi = None
        tube_contour_points = []

        # find if the tube is in position
        if tube_presence:
            try:
                # get contour related items from tag info
                min_pix = self.tag_info.extras.get(GK.min_pix.value, 10)
                offset = self.tag_info.extras.get(GK.offset.value, 10)

                tube_contour_points = tube_contour(
                    original_height,
                    original_width,
                    pixel_predictions,
                    min_pix=min_pix,
                    offset=offset,
                    tube_type=self.tag_name,
                )

                diaphragm_topi = max(left_diaphragm_topi, right_diaphragm_topi)
                all_contour_points = [i for pt_list in tube_contour_points for i in pt_list]
                tube_tip = max(all_contour_points, key=lambda t: t[1])[1]

                if (tube_tip * 512 / original_height) - 50 > (diaphragm_topi * 512 / original_height):  # in position
                    tube_position = 0
                else:  # out of position
                    tube_position = 1
                    prediction = 1

            except Exception as e:
                logging.exception(e)

        if len(tube_contour_points) > 1:
            # if there are branches, tube position is indeterminate
            tube_position = -1
            tube_contour_points = []

        transformed_points = [
            tf.transform_point(pt, rmblk_l, rmblk_u) for pt_list in tube_contour_points for pt in pt_list
        ]

        _extras = {
            GK.tube_presence.value: tube_presence,  # 0 if absent , 1 if present
            # None if absent, 0 if in position, 1 if out of position, -1 if indeterminate
            GK.tube_position.value: tube_position,
            GK.tube_tip.value: int(tube_tip) if tube_tip is not None else None,
            GK.points.value: transformed_points,  # list of list of points
            GK.diaphragm_topi.value: diaphragm_topi,
        }

        results_dict = {
            GK.score.value: scan_prob_ensembled,
            GK.prediction.value: prediction,  # prediction is 1 only if tube is out of position
            GK.extras.value: _extras,
        }

        ensembled_results = base.EnsembledResults(**results_dict)

        return ensembled_results
