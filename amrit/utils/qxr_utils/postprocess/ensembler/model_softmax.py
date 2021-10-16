from typing import Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, validator
from scipy.special import softmax

from qxr_utils.image import transforms as tf
from qxr_utils.postprocess.contour.tubes_contour import Array
from qxr_utils.tag_config.classes import PredictionTypesModel
from qxr_utils.tag_config.constants import PREDICTION_MODEL_NAMES, GeneralKeys, PredictionTypes

"""
some jargon :
scan - entire cxr image
side - 2 lungs in a cxr
patch - each lung is divided into six patches

for more details, get in touch with CXR R&D team
"""
# TODO Zoom module softmax
# TODO preds dict model


def scan_softmax(scan_output: Array, batch_size: int) -> Array:
    """classification softmax on a classification (scan) model

    Args:
            scan_output (Array): scan logits  (n_models x 2)

    Returns:
            Array: softmaxed output (batch_size x n_models x 1)
    """

    assert scan_output.shape[-1] == 2, f"scan output's last dim is of length {scan_output.shape[-1]}, it should be 2"

    no_of_models = int(scan_output.shape[0] / batch_size)
    scan_prob = softmax(scan_output, axis=1).reshape(no_of_models, -1, 2).transpose(1, 0, 2)[..., 1]
    return scan_prob


def patch_softmax(patch_output: Array, batch_size: int) -> Array:
    """patch level softmax

    Args:
            patch_output (Array): patch logits (n_models x 2 x 6)

    Returns:
            Array: softmaxed output  (batch_size x n_models x 6)
    """
    assert (
        patch_output.shape[-1] == 6
    ), f"patch output's last dim is of length {patch_output.shape[-1]}, it should be 6"
    no_of_models = int(patch_output.shape[0] / batch_size)
    patch_prob = softmax(patch_output, axis=1)
    patch_prob = _manipulate_out_shape(patch_prob, batch_size, no_of_models)[:, :, 1]
    return patch_prob


def side_softmax(side_output: Array, batch_size: int) -> Array:
    """side level softmax

    Args:
            side_output (Array): side logits (n_models x 2 x 2)

    Returns:
            Array: softmaxed output  (batch_size x n_models x 2)
    """
    assert side_output.shape[-1] == 2, f"side output's last dim is of length {side_output.shape[1]}, it should be 2"
    no_of_models = int(side_output.shape[0] / batch_size)
    side_prob = softmax(side_output, axis=1)
    side_prob = _manipulate_out_shape(side_prob, batch_size, no_of_models)[:, :, 1]
    return side_prob


def side_pixel_softmax(pixel_output: Array, batch_size: int) -> Array:
    """pixel level softmax for segmentation part of 3 stream models
    Args:
            pixel_output (Array): pixel logits per side (n_models x 2 x 2 x 960 x 320)

    Returns:
            Array: softmaxed output  (batch_size x n_models x 2 x 960 x 320)
    """
    assert (
        pixel_output.shape[2] == 2
    ), f"side pixel output's 3rd dim is of length {pixel_output.shape[2]}, it should be 2"
    no_of_models = int(pixel_output.shape[0] / batch_size)
    pixel_prob = softmax(pixel_output, axis=1)
    pixel_prob = _manipulate_out_shape(pixel_prob, batch_size, no_of_models)[:, :, 1]
    return pixel_prob


def pixel_softmax(pixel_output: Array) -> Array:
    """pixel level softmax for segmentation models

    Args:
            pixel_output (Array): pixel logits over the complete image ( n_models x
            2 x im_size x im_size (512 or 224))

    Returns:
            Array: softmaxed output (n_models x im_size x im_size (512 or 224))
    """
    assert pixel_output.ndim == 4, f"pixel output's ndim is {pixel_output.ndim}, it should be 4"
    pixel_prob = softmax(pixel_output, axis=1)[:, 1]
    return pixel_prob


def _manipulate_out_shape(out, batch_size, no_of_models):
    """Transforming ndarray (out) into desired order based on batch_size and no of models.

    Args:
        out ([ndarray]): Output from model softmax.
        batch_size ([int]): batch size.
        no_of_models ([int]): no of models for tag.

    Returns:
        Array: batch_size x n_models x out.shape[2:]
    """
    # TODO Need to vectorise
    temp_list = list()
    for i in range(0, batch_size):
        each_image_pred_res_list = list()
        for model_no in range(0, no_of_models):
            index_element = np.max((batch_size * model_no, 0))
            each_image_pred_res_list.append(out[i + index_element])
        temp_list.append(np.stack(each_image_pred_res_list))
    return np.stack(temp_list)


def modify_pixel_score_via_lung_prob(pixel_scores, lung_prob):
    """Modifying pixel score based on lung probability

    Args:
        pixel_scores ([Array]): side probability (batch_size x n_models x 2)
        lung_prob ([Array]): side probability (batch_size x n_models x 2)

    Returns:
        pixel_scores ([Array]): side probability (batch_size x n_models x 2)
    """
    # TODO Need to vectorise
    for x in range(pixel_scores.shape[0]):
        for i in range(pixel_scores.shape[1]):
            for j in range(lung_prob.shape[2]):
                pixel_scores[x][i][j] = tf.scale(pixel_scores[x][i][j]) * lung_prob[x][i][j]
    return pixel_scores


class PredictionModelSoftmax(BaseModel):
    """
    This class applies softmax on logits given out by deep learning models

    if side based model or longer side model output is out, out is a list of length 4
            out[0] - scan output of size batch_size x n_models x 2
            out[1] - side output of size batch_size x n_models x 2 x 2 (n_sides) x 1
            out[2] - patch output of size batch_size x n_models x 2 x 6 (n_patches) x 1
            out[3] - pixel output of size batch_size x n_models x 2 x 2 (n_sides) x 960 x 320

    if patch based model output is out, out is a list of length 2
            out[0] - scan output of size  batch_size x n_models x 2
            out[1] - patch output of size batch_size x n_models x 2 x 6 (n_patches) x 1

    if scan based model output is out, out is a list of length 1
            out[0] - scan output of size  batch_size x n_models x 2

    if segmenation model output is out, out is a list of length 1
            out[0] - pixel output of size  batch_size x n_models x n_classes (2 or 3) x image size
            (224 or 512) x image size

    if classification_512 model output is out, out is a list of length 1
            out[0] - scan output of batch_size x size n_models x 2
    """

    model_output: Union[List, Array]
    model_name: str
    batch_size: int
    model_type: Optional[str]

    @validator("model_type", always=True)
    def check_model_type(cls, v, values):  # pragma: no cover
        """validator for model type, sets the value to a value from
                ALLOWED_PREDICTION_TYPES

        Args:
                v : model type
                values: all the class members

        Returns:
                v
        """

        if v is not None:
            assert v in PREDICTION_MODEL_NAMES, f"model_type is {v}, which is not in PREDICTION_MODEL_NAMES"
            return v
        else:
            prediction_model_types = PredictionTypesModel(**PredictionTypes.as_dict())
            tag_type = prediction_model_types._get_tag_type(values["model_name"])
            return tag_type

    def get_scan_probability(self) -> Optional[Array]:
        """returns scan probability array

        Returns:
                Array: scan probability (batch_size x n_models x 1)
        """
        probabilities = None
        if self.model_type in [
            GeneralKeys.side_based.value,
            GeneralKeys.patch_based.value,
            GeneralKeys.scan_224.value,
            GeneralKeys.scan_320.value,
            GeneralKeys.longer_side.value,
            GeneralKeys.classification_512.value,
            GeneralKeys.scan_960.value,
        ]:
            out = self.model_output[0]
            probabilities = scan_softmax(out, self.batch_size)
        return probabilities

    def get_patch_probability(self) -> Optional[Array]:
        """returns patch probability array

        Returns:
                Array: patch probability (batch_size x n_models x 6)
        """
        probabilities = None
        if self.model_type in [GeneralKeys.side_based.value, GeneralKeys.longer_side.value]:
            out = self.model_output[2].squeeze(axis=-1)
        elif self.model_type in [GeneralKeys.patch_based.value]:
            out = self.model_output[1].squeeze(axis=-1)

        else:
            return probabilities
        probabilities = patch_softmax(out, self.batch_size)

        return probabilities

    def get_side_probability(self) -> Optional[Array]:
        """returns side probability array

        Returns:
                Array: side probability (batch_size x n_models x 2)
        """
        probabilities = None
        if self.model_type in [GeneralKeys.side_based.value, GeneralKeys.longer_side.value]:
            out = self.model_output[1].squeeze(axis=-1)
            probabilities = side_softmax(out, self.batch_size)
        return probabilities

    def get_side_pixel_probability(self) -> Array:
        """returns side pixel probability array

        Returns:
                Array: side pixel probability (batch_size x n_models x 2 x 960 x 320)
        """
        probabilities = None
        if self.model_type in [GeneralKeys.side_based.value, GeneralKeys.longer_side.value]:
            out = self.model_output[3]
            probabilities = side_pixel_softmax(out, self.batch_size)
        return probabilities

    def get_pixel_probability(self) -> Array:
        """returns pixel probability array

        Returns:
                Array: pixel probability (batch_size x n_models x 2 x im_size x im_size (512 or 224))
        """
        probabilities = None
        if self.model_type in [GeneralKeys.segmentation_224.value, GeneralKeys.segmentation_512.value]:
            out = self.model_output[0]
            probabilities = pixel_softmax(out)
        return probabilities

    def get_softmaxed_probabilities(self) -> Dict[str, Array]:
        """Returns a dict of softmaxed probabilities with the following keys
        score -> scan probabilities
        patch_scores -> patch probabilities
        side_scores -> side probabilities
        pixel_scores -> side level segmentation probabilities or full scan
        segmentation probabilities, which ever is available. Both will not
        available together

        if one of the above keys is not available, None will be its value

        Returns:
                Dict[str, Array]: [description]
        """
        side_pixel_prob = self.get_side_pixel_probability()
        pixel_prob = self.get_pixel_probability()
        pixel_scores = None
        if side_pixel_prob is not None:
            pixel_scores = side_pixel_prob
        if pixel_prob is not None:
            pixel_scores = pixel_prob
        lung_prob = self.get_side_probability()
        if self.model_type in GeneralKeys.side_based.value:
            pixel_scores = modify_pixel_score_via_lung_prob(pixel_scores, lung_prob)

        out = {
            GeneralKeys.score.value: self.get_scan_probability(),
            GeneralKeys.patch_scores.value: self.get_patch_probability(),
            GeneralKeys.side_scores.value: lung_prob,
            GeneralKeys.pixel_scores.value: pixel_scores,
        }

        return out
