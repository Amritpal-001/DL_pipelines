from typing import Any, Dict, List

from qxr_utils.tag_config.classes import PredictionModel, TagModel, UseCaseModel
from qxr_utils.tag_config.constants import (
    PREDICTION_MODEL_NAMES,
    TAG_MODEL_NAMES,
    ConstEnum,
    DependentPredictionModels,
    PredictionTypes,
)
from qxr_utils.tag_config.use_cases import BaseUseCase


def get_use_case_model(usecase: Any) -> UseCaseModel:
    """
    Converts any use case from use_cases.py to UseCaseModel
    Args:
        usecase: usecase from use_cases.py which subclasses ConstEnum

    Returns:
        UseCaseModel
    """
    assert issubclass(usecase, BaseUseCase)
    data = {"name": usecase.__name__}
    data.update(usecase.get_use_case_dict())
    # print(data)
    use_case_model = UseCaseModel(**data)
    return use_case_model
