"""
This file contains all the base classes related to ensemblers
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, validator

from qxr_utils.postprocess.contour.tubes_contour import Array
from qxr_utils.tag_config.classes import PredictionModel, TagModel, UseCaseModel
from qxr_utils.tag_config.constants import TAG_MODEL_NAMES, GeneralKeys


class EnsembledResults(BaseModel):
    score: float
    prediction: bool
    patch_scores: Optional[Array]
    patch_predictions: Optional[Array]
    side_scores: Optional[Array]
    side_predictions: Optional[Array]
    ensemble_pixel: Optional[Dict]
    extras: Optional[Dict[str, Any]]

    @validator("score")
    def check_score(cls, v):
        assert 0 <= v <= 1
        return v

    @validator("patch_scores", always=True)
    def check_patch_scores(cls, v):
        if v is not None:
            assert v.shape == (6,)
            if np.all(v == -1):  # legacy check for previous scan model, remove later
                return v
            else:
                assert np.all(v <= 1)
                assert np.all(v >= 0)
                return v

    @validator("patch_predictions", always=True)
    def check_patch_predictions(cls, v):
        if v is not None:
            assert v.shape == (6,)
            assert np.array_equal(v, v.astype(bool))
            return v

    @validator("side_scores", always=True)
    def check_side_scores(cls, v):
        if v is not None:
            assert v.shape == (2,)
            assert np.all(v <= 1)
            assert np.all(v >= 0)
            return v

    @validator("side_predictions", always=True)
    def check_side_predictions(cls, v):
        if v is not None:
            assert v.shape == (2,)
            assert np.array_equal(v, v.astype(bool))
            return v

    @validator("ensemble_pixel", always=True)
    def check_ensemble_pixel(cls, v):
        if v is not None:
            assert isinstance(v, dict)
            return v
        else:
            return v

    @validator("extras", always=True)
    def check_extras(cls, v):
        if v is not None:
            for k in v:
                assert k in GeneralKeys.key_names()
            return v


class BaseEnsembler(BaseModel, ABC):
    use_case_model: UseCaseModel
    tag_name: str
    preds_dict: Dict[str, Dict]
    tag_info: Optional[TagModel]
    prediction_models_info: Optional[Dict[str, PredictionModel]]
    ensemble_dict: Optional[Dict[str, Dict]]
    preprocessing_dict: Optional[Dict[str, Dict]]

    @validator("tag_name")
    def check_tag_name(cls, v):
        """
        checks if tag name is in TAG_MODEL_NAMES
        Args:
            v: "name"

        Returns:
            v
        """
        assert v in TAG_MODEL_NAMES, f"tag_name is {v}, which is not in TAG_MODEL_NAMES"
        return v

    @validator("preds_dict")
    def check_preds_dict(cls, v, values):
        """check if preds_dict has tag_name as a key

        Args:
            v : "preds_dict"
            values : all the class members (check pydantic docs for more)

        Returns:
            v
        """
        ucm = values["use_case_model"]
        tag_model = ucm.get_tag_model_config(values["tag_name"])
        dependent_models = tag_model.dependent_abnormality_prediction_models
        for i in dependent_models:
            assert i in v, f"{values['tag_name']} depends on {i}, but it is not available in preds_dict"
        return v

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.tag_info = self.get_tag_info()
        self.prediction_models_info = self.get_prediction_models_info()

    def get_tag_info(self) -> TagModel:
        """tag model information

        Returns:
            Dict: TagModel type dictionary. Refer to TagModel in tag_config.classes for detailed description of the
             keys in the dict
        """
        tag_info = self.use_case_model.get_tag_model_config(self.tag_name)
        return tag_info

    def get_prediction_models_info(self) -> Dict[str, PredictionModel]:
        tag_info = self.get_tag_info()
        dependent_abnormality_prediction_models = tag_info.dependent_abnormality_prediction_models

        prediction_models_info = {}
        for i in dependent_abnormality_prediction_models:
            prediction_models_info[i] = self.use_case_model.get_prediction_model_config(i)

        return prediction_models_info

    @abstractmethod
    def ensemble(self) -> EnsembledResults:
        """override this method for to implement your own ensembler.
        This should return a dict with ensembled predictions"""
        raise NotImplementedError
