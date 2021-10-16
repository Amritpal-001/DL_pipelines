import logging
from typing import Any, Dict, Tuple

import cv2
import numpy as np
from pydantic import validator
from skimage import morphology as mph

import qxr_utils.image.contour_utils as image_ci
from qxr_utils.image import transforms as tf
from qxr_utils.postprocess.ensembler import base
from qxr_utils.tag_config.classes import PredictionModel, UseCaseModel
from qxr_utils.tag_config.constants import DependentPredictionModels
from qxr_utils.tag_config.constants import GeneralKeys as GK

logger = logging.getLogger("TAG ENSEMBLER")


class TagEnsembler(base.BaseEnsembler):
    use_case_model: UseCaseModel
    tag_name: str
    preds_dict: Dict[str, Dict]
    preprocessing_dict: Dict[str, Any]

    def adjust_patch_with_pixel_preds(
        self, results_dict: Dict[str, Any], model_info: PredictionModel, scan_score: float
    ) -> Dict[str, Dict]:
        """Adjusts patch predictions based on pixel predictions

        Args:
                results_dict (Dict[str, Dict]): Results dictionary for each tag
                model_info (Dict[str, PredictionModel]): model info in dictionary format
                scan_score (float): scan score of scan prediction

        Returns:
                Dict[str, Dict]: updated results dictionary
        """
        mask_width = 320  # default
        nb_sides = 2
        nb_patches_on_each_side = 3
        if results_dict[GK.ensemble_pixel.value] is not None:
            pixel_th = model_info.pixel_threshold
            scan_th = model_info.scan_threshold

            pixel_prob = results_dict[GK.pixel_scores.value]
            pixmap = ((pixel_prob > pixel_th) * 1).sum(axis=0)
            pixmap = (pixmap > 0) * (scan_score > scan_th)

            corrected_patch_preds = []
            for side in range(nb_sides):
                for j in range(nb_patches_on_each_side):
                    h1, h2 = j * mask_width, (j + 1) * mask_width
                    percentage = np.mean(pixmap[side, h1:h2, :])
                    lung_score = results_dict[GK.ensemble_lung.value][side]
                    corrected_patch_preds.append(
                        ((percentage > 0.04) * (lung_score > scan_th) * (scan_score > scan_th)) * 1
                    )
            results_dict[GK.ensemble_patch.value] = np.array(corrected_patch_preds)
        return results_dict

    def threshold_pixel_map(self, tag: str, preds_dict: Dict[str, Dict], model_info: PredictionModel) -> np.ndarray:
        """Thresholds a pixel map with a threshold and smoothens it

        Args:
                tag (str): tag name
                preds_dict (Dict[str, Dict]): predictions dictionary
                model_info (Dict[str, PredictionModel]): model configuration as a dictionary

        Returns:
                np.ndarray: numpy array of the changed pixel map which is smoothened
        """
        pixel_prob = {}
        pixel_prob[tag] = preds_dict[tag][GK.pixel_scores.value]
        for j in range(len(pixel_prob[tag])):
            pixel_prob[tag][j] = (pixel_prob[tag][j] > model_info.pixel_threshold) * 1
        pixel_prob[tag] = np.sum(pixel_prob[tag], axis=0)

        ## Forming smooth contour for tag
        contour_area_th, contour_type = model_info.contour_area_threshold, model_info.contour_shape
        for i in range(len(pixel_prob[tag])):
            pixel_prob[tag][i] = image_ci.generate_smooth_solid_mask(pixel_prob[tag][i], contour_area_th, contour_type)

        final_map = pixel_prob[tag]

        return final_map

    def refine_pixel_map(self, tag: str, results_dict: Dict[str, Dict], model_info: PredictionModel) -> Dict:
        """Refines a pixel map, removes the diaphragm part if needed, computes area present and returns the output

        Args:
                tag (str): tag name
                results_dict (Dict[str, Dict]): results dictionary
                model_info (Dict[str, PredictionModel]): model info dictionary

        Returns:
                Dict: dictionary containing area, pixel info etc.
        """
        tag_threshold = model_info.scan_threshold

        pixel_map = self.threshold_pixel_map(tag, self.preds_dict, model_info)
        lung_masks = self.preprocessing_dict[GK.lung_masks.value]
        fs_nparray = self.preprocessing_dict[GK.fsnparray.value]
        lung_areas = self.preprocessing_dict[GK.lung_areas.value]
        bottom_tags = [x.value for x in [GK.pneumoperitoneum, GK.pleuraleffusion, GK.bluntedcp]]
        if tag not in bottom_tags:
            left_pixel = tf.cast_uint8((((pixel_map[0] > 0) * 1 + lung_masks[0]) > 1) * 1)
            right_pixel = tf.cast_uint8((((pixel_map[1] > 0) * 1 + lung_masks[1]) > 1) * 1)
            sb = self.preprocessing_dict[GK.side_borders.value]
        else:
            left_pixel = tf.cast_uint8((pixel_map[0] > 0) * 1)
            right_pixel = tf.cast_uint8((pixel_map[1] > 0) * 1)
            sb = self.preprocessing_dict[GK.side_borders_bottom.value]

        left_pixel = cv2.resize(left_pixel, (sb[0][3] - sb[0][2], sb[0][1] - sb[0][0]))
        right_pixel = cv2.resize(right_pixel, (sb[1][3] - sb[1][2], sb[1][1] - sb[1][0]))

        left_contour, left_abn_area, _ = image_ci.find_contour(left_pixel, solid_mask=True)
        right_contour, right_abn_area, _ = image_ci.find_contour(right_pixel, solid_mask=True)

        left_contour_full = np.zeros(np.shape(fs_nparray), dtype=np.uint8)
        right_contour_full = np.zeros(np.shape(fs_nparray), dtype=np.uint8)

        scan_probability = self.preds_dict[tag][GK.ensemble_scan.value]

        ## Left Area is the complete area of left lung in pixels.
        ## Not the area of the abnormality which is present
        left_area = (sb[0][3] - sb[0][2]) * (sb[0][1] - sb[0][0]) * lung_areas[0]
        right_area = (sb[1][3] - sb[1][2]) * (sb[1][1] - sb[1][0]) * lung_areas[1]

        left_abn_area_percent = round(100 * (left_abn_area / left_area), 1)
        right_abn_area_percent = round(100 * (right_abn_area / right_area), 1)

        if scan_probability > tag_threshold:
            left_contour_full[sb[0][0] : sb[0][1], sb[0][2] : sb[0][3]] = left_contour
            right_contour_full[sb[1][0] : sb[1][1], sb[1][2] : sb[1][3]] = right_contour

        left_contour_full = (left_contour_full > 0).astype(bool)
        right_contour_full = (right_contour_full > 0).astype(bool)

        left_contour_full = tf.np_compress(left_contour_full)
        right_contour_full = tf.np_compress(right_contour_full)

        left_contour = tf.np_compress(cv2.resize(left_contour, (320, 960), cv2.INTER_AREA))
        right_contour = tf.np_compress(cv2.resize(right_contour, (320, 960), cv2.INTER_AREA))

        out = {
            "mask": (left_contour_full, right_contour_full),
            "mask_side": (left_contour, right_contour),
            "area": (left_area, right_area),
            "abn_area_percent": (left_abn_area_percent, right_abn_area_percent),
        }
        return out

    def ensemble(self) -> base.EnsembledResults:
        """Main ensemble function inherited from base class

        Returns:
                base.EnsembledResults: ensembled results for individual tags
        """
        prediction_model_info = self.prediction_models_info[self.tag_name]
        ensembled_scan_score = self.preds_dict[self.tag_name][GK.ensemble_scan.value]
        ensembled_patch_scores = self.preds_dict[self.tag_name][GK.ensemble_patch.value]
        if GK.ensemble_pixel.value in self.preds_dict[self.tag_name]:
            self.preds_dict[self.tag_name] = self.adjust_patch_with_pixel_preds(
                self.preds_dict[self.tag_name], prediction_model_info, ensembled_scan_score
            )
            ensembled_patch_scores = self.preds_dict[self.tag_name][GK.ensemble_patch.value]
        ensembled_scan_prediction = ensembled_scan_score > prediction_model_info.scan_threshold

        if not ensembled_scan_prediction or ensembled_patch_scores is None:
            ensembled_patch_predictions = np.array([False, False, False, False, False, False])
        else:
            ensembled_patch_predictions = ensembled_patch_scores > prediction_model_info.patch_threshold
            if ensembled_patch_predictions.sum() == 0:
                ensembled_patch_predictions[ensembled_patch_scores.argmax()] = True

        out = {
            GK.score.value: ensembled_scan_score,
            GK.prediction.value: ensembled_scan_prediction,
            GK.patch_scores.value: ensembled_patch_scores,
            GK.patch_predictions.value: ensembled_patch_predictions,
        }

        if GK.ensemble_pixel.value in self.preds_dict[self.tag_name]:
            out[GK.extras.value] = {GK.pixel_scores.value: self.preds_dict[self.tag_name][GK.pixel_scores.value]}
            if out[GK.extras.value][GK.pixel_scores.value] is not None:
                out[GK.ensemble_pixel.value] = self.refine_pixel_map(
                    self.tag_name, self.preds_dict, prediction_model_info
                )
        ensembled_results = base.EnsembledResults(**out)
        return ensembled_results


class AbnormalEnsembler(base.BaseEnsembler):
    use_case_model: UseCaseModel
    tag_name = GK.abnormal.value
    preds_dict: Dict[str, Dict]

    def which_one_of_three(self) -> str:
        """classified the abnormality as normal/abnormal/toberead

        Returns:
                str: GK.normal/GK.abnormal/GK.toberead
        """
        abnormal_tags = DependentPredictionModels.abnormal.value

        def sure_abnormal(tag):
            model_info = self.prediction_models_info[tag]
            return self.preds_dict[tag][GK.score.value] > model_info.abnormal_threshold

        def sure_normal(tag):
            model_info = self.prediction_models_info[tag]
            return self.preds_dict[tag][GK.score.value] < model_info.normal_threshold

        bucket = GK.toberead.value
        tags_list = []
        for abnormal_tag in abnormal_tags:
            if abnormal_tag in self.preds_dict:
                tags_list.append(abnormal_tag)
        if all(map(sure_normal, tags_list)):
            bucket = GK.normal.value
        elif any(map(sure_abnormal, tags_list)):
            bucket = GK.abnormal.value

        return bucket

    def ensemble(self) -> base.EnsembledResults:
        """Abnormal Ensemble model

        Returns:
                base.EnsembledResults: ensembled results
        """
        abnormal_tags = DependentPredictionModels.abnormal.value
        preds_list = []
        scores_list = []
        for abnormal_tag in abnormal_tags:
            if abnormal_tag in self.preds_dict:
                preds_list.append(self.preds_dict[abnormal_tag][GK.prediction.value])
                scores_list.append(self.preds_dict[abnormal_tag][GK.score.value])
        scan_prediction = any(preds_list)
        if scan_prediction:
            scan_score = np.array(scores_list)[preds_list].mean()
        else:
            scan_score = np.array(scores_list).mean()

        nva_class = self.which_one_of_three()

        out = {
            GK.score.value: scan_score,
            GK.prediction.value: scan_prediction,
            GK.patch_scores.value: np.array([-1, -1, -1, -1, -1, -1]),
            GK.patch_predictions.value: np.array([False, False, False, False, False, False]),
            GK.extras.value: {GK.nva_class.value: nva_class},
        }

        abnormal_results = base.EnsembledResults(**out)

        return abnormal_results


class CovidEnsembler(base.BaseEnsembler):
    use_case_model: UseCaseModel
    tag_name = GK.covid.value
    preds_dict: Dict[str, Dict]
    preprocessing_dict: Dict[str, Any]

    def get_heatmap_left_right(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns left and right opacity + consolidation heatmaps

        Returns:
                Tuple[np.ndarray, np.ndarray]: left heatmap, right heatmap return
        """
        opacity_preds = self.preds_dict[GK.opacity.value][GK.ensemble_pixel.value]["mask_side"]
        consolidation_preds = self.preds_dict[GK.consolidation.value][GK.ensemble_pixel.value]["mask_side"]
        hm_left = tf.np_decompress(opacity_preds[0]) + tf.np_decompress(consolidation_preds[0])
        hm_right = tf.np_decompress(opacity_preds[1]) + tf.np_decompress(consolidation_preds[1])
        return hm_left, hm_right

    def periphery_percent(self, hm: np.ndarray, lung_mask: np.ndarray, side: str = "right") -> Tuple[float, float]:
        """Returns the periphery area affected

        Args:
                hm (np.ndarray): abnormal heatmap
                lung_mask (np.ndarray): lung mask heatmap
                side (str, optional): which side to calculate periphery for. Defaults to "right".

        Returns:
                float, float: periphery area affected
        """
        lung_mask = cv2.resize(lung_mask * 1.0, (40, 120))
        lung_mask = (lung_mask > 0) * 1
        canvas = np.zeros((lung_mask.shape[0] + 2, lung_mask.shape[1] + 2))
        canvas[1:-1, 1:-1] = lung_mask
        if side == "right":
            canvas[1:-1, -1 * int(canvas.shape[1] / 6) :] = canvas.max()
        else:
            canvas[1:-1, : int(canvas.shape[1] / 6)] = canvas.max()
        hm = (cv2.resize(hm * 1.0, (canvas.shape[1], canvas.shape[0])) > 0) * 1.0
        affected = hm.sum() / lung_mask.sum() * 1.5
        im_erode = mph.erosion(canvas, mph.disk(12))
        periphery = canvas - im_erode
        if hm.sum() > 0:
            periphery_percentage = (hm * periphery).sum() / hm.sum()
        else:
            periphery_percentage = 0
        return periphery_percentage, affected

    def lower_lobe(self, hm: np.ndarray) -> float:
        """percentage affected for lower lobe

        Args:
                hm (np.ndarray): heatmap of a particular lung

        Returns:
                float: percentage affected for that particular lower lobe
        """
        h = hm.shape[0]
        lower_mask = hm[int(2 * h / 3) : h, :]
        return lower_mask.sum() / (hm.sum() + 0.000000001)

    def dice(self, a: np.ndarray, b: np.ndarray) -> float:
        """Function to calculate dice score

        Args:
                a (np.ndarray): first array input
                b (np.ndarray): second array input

        Returns:
                float: value of dice score in float
        """
        return 2 * (a * b).sum() / (a.sum() + b.sum() + 0.00000001)

    def ensemble(self) -> base.EnsembledResults:
        """covid ensemble final function. This function should be run after abnormal ensemble has been run

        Returns:
                base.EnsembledResults: output converted to base ensemble format
        """
        hm_left, hm_right = self.get_heatmap_left_right()
        opacity_score = max(
            self.preds_dict[GK.opacity.value][GK.score.value],
            self.preds_dict[GK.consolidation.value][GK.score.value],
        )
        thorax_mask_left, thorax_mask_right = self.preprocessing_dict["lung_masks"]
        pph_percent_left, pph_percent_right = 0.0, 0.0
        affected_left, affected_right = 0.0, 0.0
        if hm_left.max() > 0:
            pph_percent_left, affected_left = self.periphery_percent(hm_left, thorax_mask_left, "left")
        if hm_right.max() > 0:
            pph_percent_right, affected_right = self.periphery_percent(hm_right, thorax_mask_right, "right")

        pph = max(0, pph_percent_left, pph_percent_right)

        affected_score = (affected_left + affected_right) / 2
        bilateral = (hm_left.max() > 0) and (hm_right.max() > 0)
        bilateral_score = self.dice(hm_left, np.fliplr(hm_right)) * bilateral
        lowerlobe_score = max(self.lower_lobe(hm_left), self.lower_lobe(hm_right))

        # peffusion_score = self.preds_dict[GK.pleuraleffusion.value][GK.score.value]
        # consolidation_score = self.preds_dict[GK.consolidation.value][GK.score.value]
        cavity_score = self.preds_dict[GK.cavity.value][GK.score.value]

        contra_indication_score = -0.5 * (cavity_score > 0.75)

        ncov_smooth_score = (
            opacity_score
            + bilateral_score
            + 0.5 * lowerlobe_score
            + affected_score * pph * 2
            + contra_indication_score
        )

        consistency_with_covid_findings = GK.covid_none.value

        if 1 < ncov_smooth_score < 1.6:
            consistency_with_covid_findings = GK.covid_low.value
        elif 1.6 <= ncov_smooth_score <= 2.1:
            consistency_with_covid_findings = GK.covid_medium.value
        elif ncov_smooth_score > 2.1:
            consistency_with_covid_findings = GK.covid_high.value
        elif ncov_smooth_score < 1:
            consistency_with_covid_findings = GK.covid_none.value

        percentage = round(90 * min(ncov_smooth_score / 3, 1), 1)

        nva_class = self.preds_dict[GK.abnormal.value][GK.extras.value][GK.nva_class.value]

        if nva_class == GK.normal.value:
            consistency_with_covid_findings = GK.covid_none.value
            percentage = 0.0
        elif nva_class == GK.toberead.value:
            consistency_with_covid_findings = GK.covid_na.value
            percentage = 0.0
        covid_score = percentage / 100
        covid_prediction = consistency_with_covid_findings in [GK.covid_high.value, GK.covid_medium.value]
        extras = {GK.covid_percentage.value: percentage, GK.covid_risk.value: consistency_with_covid_findings}
        output = {GK.score.value: covid_score, GK.prediction.value: covid_prediction, GK.extras.value: extras}
        covid_results = base.EnsembledResults(**output)
        return covid_results
