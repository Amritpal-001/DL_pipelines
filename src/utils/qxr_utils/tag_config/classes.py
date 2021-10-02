from typing import Any, Dict, List, Optional

from pydantic import BaseModel, root_validator, validator

from qxr_utils.tag_config.constants import (
    ALLOWED_PREDICTION_TYPES,
    CONTOUR_NAMES_COMBINED,
    CONTOUR_SHAPES,
    PREDICTION_MODEL_NAMES,
    TAG_MODEL_NAMES,
    USE_CASES,
    ContourShapes,
    DependentPredictionModels,
    GeneralKeys,
    PredictionTypes,
)


# noinspection PyUnresolvedReferences
class PredictionModel(BaseModel):
    """
    Any deep learning model, it has the below mentioned attributes
    #TODO combined validator for tag_name and tag_type
    #TODO use allow_reuse style validators - https://pydantic-docs.helpmanual.io/usage/validators/#reuse-validators
    Attributes:
        name : name of the prediction model
        scan_threshold: scan level threshold
        patch_threshold: patch level threshold
        pixel_threshold: pixel level threshold
        model_type: type of prediction model
        to_run: if the model should be run for inference
        extras: any additional keys related to a prediction model
    """

    name: str
    scan_threshold: float = 0.5
    patch_threshold: List[float] = [0.5] * 6
    pixel_threshold: float = 0.5
    normal_threshold: float = 0.5
    abnormal_threshold: float = 0.5
    contour_area_threshold: int = 2000
    contour_shape: str = "convex"
    model_type: str
    to_run: bool = True
    extras: Optional[Dict[str, Any]]

    @validator("scan_threshold", "pixel_threshold")
    def check_scan_pixel_threshold(cls, v):
        """
        checks if the threshold is between 0 and 1
        Args:
            v: one of "scan_threshold",  "pixel_threshold"

        Returns:
            v
        """
        assert 0 <= v <= 1, f"scan or pixel threshold is not in between 0 and 1, it is {v}"
        return v

    @validator("contour_area_threshold")
    def check_contour_area_threshold(cls, v):
        """
        checks if the threshold is between 0 and 1
        Args:
            v: one of "scan_threshold",  "pixel_threshold"

        Returns:
            v
        """
        assert v >= 900
        return v

    @validator("contour_shape")
    def check_contour_shape(cls, v):
        """
        checks if the threshold is between 0 and 1
        Args:
            v: one of "scan_threshold",  "pixel_threshold"

        Returns:
            v
        """
        assert v == "concave" or v == "convex"
        return v

    @validator("patch_threshold")
    def check_patch_threshold(cls, v):
        """
        checks if each item of the patch threshold is between 0 and 1
        Args:
            v: "patch_threshold"

        Returns:
            v
        """
        assert len(v) == 6
        for i in v:
            assert 0 <= i <= 1, f"one of the path thresholds is not in between 0 and 1, it is {i}"
        return v

    @validator("name")
    def check_tag_name(cls, v):
        """
        checks if the tag name is in PREDICTION_MODEL_NAMES
        Args:
            v: "name"

        Returns:
            v
        """
        assert v in PREDICTION_MODEL_NAMES, f"name is {v}, which is not in PREDICTION_MODEL_NAMES"
        return v

    @validator("model_type")
    def check_tag_type(cls, v):
        """
        checks if model type is in ALLOWED_PREDICTION_TYPES
        Args:
            v: "model_type"

        Returns:
            v
        """
        assert v in ALLOWED_PREDICTION_TYPES, f"model_type is {v}, which is not in ALLOWED_PREDICTION_TYPES"
        return v

    @validator("extras", always=True)
    def check_extras(cls, v):
        if v is not None:
            for k in v:
                assert k in GeneralKeys.key_names(), f"extras has {k}, which is not in GeneralKeys"
            return v


# noinspection PyUnresolvedReferences
class TagModel(BaseModel):
    """
    Any tag (such as opacity, abnormal), it can have dependent prediction models
    Attributes:
        name: name of the tag
        dependent_abnormality_prediction_models: list of prediction models the tag needs
        normal_threshold: threshold below (<=) which it is normal
        abnormal_threshold: threshold above (>=) which it is abnormal
        to_run: if the tag and dependent_abnormality_prediction_models needs to be run
        to_show: if the tag results needs to be shown in the final output such as secondary capture, report
        has_contour: if the tag has contour
        contour_shape: shape of the contour [concave, convex]
        show_contour: if the contour should be shown in the secondary capture
        extras: any additional keys related to a tag model
    """

    name: str
    dependent_abnormality_prediction_models: List[str]
    normal_threshold: float = 0.5
    abnormal_threshold: float = 0.5
    to_run: bool = True
    to_show: bool = True
    has_contour: bool
    contour_area_threshold: int = 2000
    contour_shape: str = "convex"
    show_contour: bool = True
    extras: Optional[Dict[str, Any]]

    @validator("name")
    def check_name(cls, v):
        """
        checks if tag name is in TAG_MODEL_NAMES
        Args:
            v: "name"

        Returns:
            v
        """
        assert v in TAG_MODEL_NAMES, f"name is {v}, which is not in TAG_MODEL_NAMES"
        return v

    @validator("contour_shape")
    def check_contour_type(cls, v, values):
        """
        checks if has_contour is True, contour type is in CONTOUR_SHAPES
        Args:
            v: "contour_shape"
            values: all the class members (check pydantic docs for more)

        Returns:
            v
        """
        if values["has_contour"]:
            assert v in CONTOUR_SHAPES, f"contour_shape is {v}, which is not in CONTOUR_SHAPES"
        return v

    @validator("normal_threshold")
    def check_normal_threshold(cls, v):
        """
        checks if normal threshold is between 0,1
        Args:
            v: "normal_threshold"

        Returns:
            v
        """
        assert 0 <= v <= 1, f"normal_threshold should be between 0 and 1, it is {v}"
        return v

    @validator("abnormal_threshold")
    def check_abnormal_threshold(cls, v, values):
        """
        checks if normal threshold is between 0,1 and if normal th <= abnormal th
        Args:
            v: "abnormal_threshold
            values: all the class members (check pydantic docs for more)

        Returns:
            v
        """
        assert 0 <= v <= 1, f"abnormal_threshold should be between 0 and 1, it is {v}"
        assert values["normal_threshold"] <= v, (
            f"abnormal_threshold should be > normal threshold, where as normal_threshold is "
            f"{values['normal_threshold']} and abnormal threshold is {v}"
        )
        return v

    @validator("dependent_abnormality_prediction_models", each_item=True)
    def check_dependent_prediction_models(cls, v):
        """
        checks if each of dependent_abnormality_prediction_models is in PREDICTION_MODEL_NAMES
        Args:
            v: each item of dependent_abnormality_prediction_models

        Returns:
            v
        """
        assert (
            v in PREDICTION_MODEL_NAMES
        ), f"dependent_abnormality_prediction_models has {v}, which is  not in PREDICTION_MODEL_NAMES"
        return v

    @validator("show_contour")
    def check_show_contour(cls, v, values):
        """
        sets show_contour to false if to_show is false
        Args:
            v: "show_contour"
            values: all the class members (check pydantic docs for more)

        Returns:
            v
        """
        if not values["to_show"]:
            v = False
        return v

    @validator("to_show")
    def check_to_show(cls, v, values):
        """
        set to_show to false if to_run is false
        Args:
            v: "to_show"
            values: all the class members (check pydantic docs for more)

        Returns:
            v
        """
        if not values["to_run"]:
            v = False
        return v

    @validator("extras", always=True)
    def check_extras(cls, v):
        if v is not None:
            for k in v:
                assert k in GeneralKeys.key_names(), f"extras has {k}, which is not in GeneralKeys"
            return v

    @staticmethod
    def get_dependent_abnormality_models(tag_name: str) -> List[str]:
        """
        self explanatory
        Args:
            tag_name: tag name

        Returns:
            list of dependent abnormality models
        """
        if tag_name in DependentPredictionModels.key_names():
            dep_models = getattr(DependentPredictionModels, tag_name)
            return dep_models.value
        else:
            return [tag_name]


# noinspection PyUnresolvedReferences
class UseCaseModel(BaseModel):
    """
    use case and dependent models
    #TODO add output types and sc related stuff
    Attributes:
        name: name of the use case
        tags_to_show: list of tag results to be shown in the final output
        tags_to_run: list of tags to be run before showing the final output
        tags_to_combine: list of tags to be combined and shown as a contour
        #TODO set defaults for normal, abnormal, scan, patch, pixel ths in constants
        normal_thresholds: dict of tag name, normal threshold, if this is none, default will be 0.5
        abnormal_thresholds: dict of tag name, abnormal threshold, if this is none, default will be 0.5
        contours_to_show: dict of tag name, bool (true if the contour has to be shown), if this is none,
        value will be picked from ContourShapes in constants
        scan_thresholds: dict of prediction models and scan thresholds
        patch_thresholds: dict of prediction models and patch thresholds
        pixel_thresholds: dict of prediction models and pixel thresholds
        extras: any additional keys related to a tag models or prediction models
    """

    name: str
    # for tag models
    # TODO convert List[str] to List[TagModel]
    tags_to_show: List[str]
    tags_to_run: List[str]
    tags_to_combine: Dict[str, List[str]]
    normal_thresholds: Optional[Dict[str, float]]
    abnormal_thresholds: Optional[Dict[str, float]]
    contours_to_show: Optional[Dict[str, bool]]
    dependent_prediction_models: Optional[Dict[str, List[str]]]
    # for prediction models
    scan_thresholds: Optional[Dict[str, float]]
    patch_thresholds: Optional[Dict[str, List[float]]]
    pixel_thresholds: Optional[Dict[str, float]]
    extras: Optional[Dict[str, Any]]

    @validator("name")
    def check_name(cls, v):
        """
        check if name is in use_cases
        Args:
            v: "name"

        Returns:
            v
        """
        assert v in USE_CASES, f"name is {v}, which is not in USE_CASES"
        return v

    @validator("tags_to_run")
    def check_tags_to_run(cls, v):
        """
        check if each item of tags_to_run are in TAG_MODEL_NAMES
        Args:
            v: each item of "tags_to_run"

        Returns:
            v
        """
        for tag in v:
            assert tag in TAG_MODEL_NAMES, f"tags_to_run has {tag}, which is not in TAG_MODEL_NAMES"
        return v

    @validator("tags_to_show")
    def check_tags_to_show(cls, v):
        """
        check if each item of tags_to_show are in TAG_MODEL_NAMES
        Args:
            v: each item of "tags_to_show"

        Returns:
            v
        """
        for tag in v:
            assert tag in TAG_MODEL_NAMES, f"tags_to_show has {tag}, which is not in TAG_MODEL_NAMES"
        return v

    @validator("tags_to_combine")
    def check_tags_to_combine(cls, v):
        """
        check if each key of tags_to_combine is in CONTOUR_NAMES_COMBINED
        and if each element in value is in TAG_MODEL_NAMES
        Args:
            v: "tags_to_combine"

        Returns:
            v
        """
        for combined_tag, tags_list in v.items():
            assert (
                combined_tag in CONTOUR_NAMES_COMBINED
            ), f"tags_to_combine has {combined_tag} as a key, which is not in CONTOUR_NAMES_COMBINED"
            for tag_name in tags_list:
                assert (
                    tag_name in TAG_MODEL_NAMES
                ), f"tags_to_combine has {tag_name} in its values, which not in TAG_MODEL_NAMES"
        return v

    @validator("normal_thresholds", "abnormal_thresholds", always=True)
    def check_normal_abnormal_thresholds(cls, v, values):
        """
        validators for thresholds
        if v is none, then it sets the default values of thresholds to 0.5
        Args:
            v: "normal_thresholds","abnormal_thresholds"
            values: all the class members (check pydantic docs for more)

        Returns:
            v or default values
        """
        if v is not None:
            for tag, th in v.items():
                assert tag in TAG_MODEL_NAMES, f"normal/abnormal thresholds has {tag}, which is not in TAG_MODEL_NAMES"
                assert 0 <= th <= 1, f"normal/abnormal thresholds for {tag} not in between 0 and 1, it is {th}"
            return v

        else:
            default_ths = {}
            for tag_name in values["tags_to_run"]:
                default_ths[tag_name] = 0.5
            return default_ths

    @validator("scan_thresholds", "pixel_thresholds", always=True)
    def check_scan_pixel_thresholds(cls, v):
        """
        validators for thresholds
        if v is none, then it sets the default values of thresholds to 0.5
        Args:
            v: "scan_thresholds","pixel_thresholds"
            values: all the class members (check pydantic docs for more)

        Returns:
            v or default values
        """
        if v is not None:
            for tag, th in v.items():
                assert (
                    tag in PREDICTION_MODEL_NAMES
                ), f"scan/pixel thresholds has {tag}, which is not in PREDICTION_MODEL_NAMES"
                assert 0 <= th <= 1, f"scan/pixel thresholds for {tag} not in between 0 and 1, it is {th}"
            return v

        else:
            default_ths = {}
            for tag_name in PREDICTION_MODEL_NAMES:
                default_ths[tag_name] = 0.5
            return default_ths

    @validator("patch_thresholds", always=True)
    def check_patch_thresholds(cls, v):
        """
        validators for thresholds
        if v is none, then it sets the default values of thresholds to 0.5
        Args:
            v: "patch_thresholds"
            values: all the class members (check pydantic docs for more)

        Returns:
            v or default values
        """
        if v is not None:
            for tag, th in v.items():
                assert (
                    tag in PREDICTION_MODEL_NAMES
                ), f"patch_thresholds has {tag}, which is not in PREDICTION_MODEL_NAMES"
                for i in th:
                    assert 0 <= i <= 1, f"patch thresholds for {tag} not in between 0 and 1, it is {th}"
            return v

        else:
            default_ths = {}
            for tag_name in PREDICTION_MODEL_NAMES:
                default_ths[tag_name] = [0.5] * 6
            return default_ths

    @validator("dependent_prediction_models", always=True)
    def check_dependent_prediction_models(cls, v, values):
        """
        validators for dependent_prediction_models
        if v is none, then it sets the default value from DependentPredictionModels via TagModel
        Args:
            v: "dependent_prediction_models"
            values: all the class members (check pydantic docs for more)

        Returns:
            v or default values
        """
        if v is not None:
            for tag, models in v.items():
                assert (
                    tag in TAG_MODEL_NAMES
                ), f"dependent_prediction_models has {tag} in the keys, which is not in TAG_MODEL_NAMES"
                for m in models:
                    assert (
                        m in PREDICTION_MODEL_NAMES
                    ), f"dependent_prediction_models has {m} in the values, which is not in PREDICTION_MODEL_NAMES"
            return v
        else:
            dependent_prediction_models = {
                tag: TagModel.get_dependent_abnormality_models(tag) for tag in values["tags_to_run"]
            }
            return dependent_prediction_models

    @validator("contours_to_show", always=True)
    def check_contours_to_show(cls, v, values):
        """
        validators for contours_to_show
        if v is none, then it sets the default value from ContourShapes
        Args:
            v: "contours_to_show"
            values: all the class members (check pydantic docs for more)

        Returns:
            v or default values
        """
        if v is not None:
            for tag in v:
                assert tag in TAG_MODEL_NAMES, f"contours_to_show has {tag} it the keys, which not in TAG_MODEL_NAMES"
            return v
        else:
            cts = {}
            for tag in values["tags_to_run"]:
                cts[tag] = tag not in ContourShapes.no_contour.value
            return cts

    @validator("extras", always=True)
    def check_extras(cls, v):
        allowed_keys = [GeneralKeys.tag_models.value, GeneralKeys.prediction_models.value]
        if v is not None:
            """
            extras = {
                'tag_models':{
                    'tuberculosis':{
                        k1:v1,...
                    }
                },
                'prediction_models':{
                    'opacity':{
                        k2:v2,...
                    }
                }

            }
            """
            for k in v:
                assert (
                    k in allowed_keys
                ), f"extras has {k} in its keys, which is not allowed. Allowed keys are {allowed_keys}"
            tag_extras = v[GeneralKeys.tag_models.value]
            prediction_extras = v[GeneralKeys.prediction_models.value]

            for _k, _v in tag_extras:
                assert (
                    _k in TAG_MODEL_NAMES
                ), f"tag_models extras has {_k} in the keys, which is not in TAG_MODEL_NAMES"
                for _k1 in _v:
                    assert (
                        _k1 in GeneralKeys.key_names()
                    ), f"tag_models values has {_k1} in the keys, which is not in GeneralKeys"

            for _k, _v in prediction_extras:
                assert (
                    _k in PREDICTION_MODEL_NAMES
                ), f"prediction_models has {_k} in its keys, which is not in PREDICTION_MODEL_NAMES"
                for _k1 in _v:
                    assert (
                        _k1 in GeneralKeys.key_names()
                    ), f"prediction_models has {_k1} in its keys, which is not in GeneralKeys"

            return v
        else:
            v = {}
            for k in allowed_keys:
                v[k] = {}
            return v

    def get_tag_model_config(self, tag: str) -> TagModel:
        """tag model config

        Args:
            tag (str): tag name

        Returns:
            Dict: TagModel type dictionary. Refer to TagModel for detailed description of the keys in the dict
        """
        # tag_models_config = {}
        contour_shapes_model = ContourShapesModel(**ContourShapes.as_dict())
        # for tag in self.tags_to_run:
        default_dependent_prediction_models = TagModel.get_dependent_abnormality_models(tag)
        tag_data = {
            GeneralKeys.name.value: tag,
            GeneralKeys.dependent_abnormality_prediction_models.value: getattr(
                self.dependent_prediction_models, tag, default_dependent_prediction_models
            ),
            GeneralKeys.normal_threshold.value: self.normal_thresholds.get(tag),
            GeneralKeys.abnormal_threshold.value: self.abnormal_thresholds.get(tag),
            GeneralKeys.to_run.value: True,
            GeneralKeys.to_show.value: tag in self.tags_to_show,
            GeneralKeys.has_contour.value: contour_shapes_model._get_has_contour(tag),
            GeneralKeys.contour_shape.value: contour_shapes_model._get_contour_type(tag),
            GeneralKeys.show_contour.value: self.contours_to_show.get(tag, False),
            GeneralKeys.extras.value: self.extras[GeneralKeys.tag_models.value].get(tag, {}),
        }
        tag_model = TagModel(**tag_data)
        # tag_models_config[tag] = tag_model.dict()

        return tag_model

    def get_prediction_model_config(self, prediction_model_name: str) -> PredictionModel:
        """prediction model config

        Args:
            prediction_model_name (str): name of the prediction model

        Returns:
            Dict: prediction model config as dict, refer to PredictionModel for detailed description of the dict keys
        """
        prediction_model_types = PredictionTypesModel(**PredictionTypes.as_dict())
        prediction_model_data = {
            GeneralKeys.name.value: prediction_model_name,
            GeneralKeys.scan_threshold.value: self.scan_thresholds.get(prediction_model_name, 0.5),
            GeneralKeys.patch_threshold.value: self.patch_thresholds.get(prediction_model_name, [0.5] * 6),
            GeneralKeys.pixel_threshold.value: self.pixel_thresholds.get(prediction_model_name, 0.5),
            GeneralKeys.model_type.value: prediction_model_types._get_tag_type(prediction_model_name),
            GeneralKeys.to_run.value: True,
            GeneralKeys.extras.value: self.extras[GeneralKeys.prediction_models.value].get(prediction_model_name, {}),
        }
        prediction_model = PredictionModel(**prediction_model_data)

        return prediction_model

    def get_prediction_models_to_run(self) -> List:
        """returns list of prediction models to run based on the config

        Returns:
            List: list of prediction models
        """

        tag_model_config = {tag: self.get_tag_model_config(tag).dict() for tag in self.tags_to_run}
        prediction_models_to_run = []
        for k, v in tag_model_config.items():
            prediction_models_to_run += v[GeneralKeys.dependent_abnormality_prediction_models.value]
        prediction_models_to_run = list(set(prediction_models_to_run))

        return prediction_models_to_run

    def generate_dict(self) -> Dict:
        """
        Outputs the config as dict in a different format
        Returns:
            Config as dict with the keys tag_models, tags_to_combine, prediction_models
        """
        tag_model_config = {tag: self.get_tag_model_config(tag).dict() for tag in self.tags_to_run}
        prediction_models_to_run = self.get_prediction_models_to_run()
        config = {
            "tag_models": tag_model_config,
            "tags_to_combine": self.tags_to_combine,
            "prediction_models_to_run": prediction_models_to_run,
        }

        config["prediction_models"] = {
            model_name: self.get_prediction_model_config(model_name).dict() for model_name in prediction_models_to_run
        }

        return config


# noinspection PyUnresolvedReferences
class PredictionTypesModel(BaseModel):
    """
    Using this to validate the PredictionTypes in constants
    Attributes:
        side_based: side based models
        patch_based: patch based models
        scan_224: scan models trained on 224x224
        scan_320: scan models trained on 320x320
        segmentation_224: segmentation models trained on 224x224
        segmentation_512: segmentation models trained on 512x512
        classification_512: classification models trained 512x512
        scan_960: scan models trained on 960x960
    """

    side_based: List[str]
    patch_based: List[str]
    scan_224: List[str]
    scan_320: List[str]
    segmentation_224: List[str]
    segmentation_512: List[str]
    classification_512: List[str]
    scan_960: List[str]

    @root_validator(pre=True)
    def check_tag_types(cls, values):
        """
        validates all the values together
        Args:
            values: all the class members

        Returns:
            values
        """
        for k in values:
            assert k in ALLOWED_PREDICTION_TYPES, f"one of the keys is {k}, which is not in ALLOWED_PREDICTION_TYPES"
        return values

    @validator("*", each_item=True)
    def check_tag_names(cls, v):
        """
        validates if each item in the list of tags in the class member is a prediction model
        Args:
            v: each item in the class member

        Returns:
            v
        """
        assert v in PREDICTION_MODEL_NAMES, f"one of the values has {v} in it, which is not in PREDICTION_MODEL_NAMES"
        return v

    def _get_tag_type(self, tag_name: str) -> str:
        """
        return tag type corresponding to input tag_name
        Args:
            tag_name: name of the tag

        Returns:
            corresponding tag type
        """
        assert tag_name in PREDICTION_MODEL_NAMES, f"tag_name is {tag_name}, which is not in PREDICTION_MODEL_NAMES"
        tag_types = list(self.__fields__.keys())
        for i in tag_types:
            if tag_name in getattr(self, i):
                return i
        return "unknown"


# noinspection PyUnresolvedReferences
class ContourShapesModel(BaseModel):
    """
    Using this to validate the ContourShapes in constants
    Attributes:
        convex: tags with convex contours
        concave: tags with concave contours
        lines: tags with line contours
        no_contour: tags with no contours
    """

    convex: List[str]
    concave: List[str]
    line: List[str]
    no_contour: List[str]

    @root_validator(pre=True)
    def check_contour_shapes(cls, values):
        """
        validates all the values combined
        Args:
            values: all the class members

        Returns:
            values
        """
        for k in values:
            assert (
                k in CONTOUR_SHAPES
            ), f"{k} in the keys, which is not a valid contour shape. Valid shapes are {CONTOUR_SHAPES}"
        return values

    @validator("*", each_item=True)
    def check_tag_names(cls, v):
        """
        validates if each item in the list of tags in the class member is a tag model
        Args:
            v: each item in the class member

        Returns:
            v
        """
        assert v in TAG_MODEL_NAMES, f"one the values has {v}, which is not in TAG_MODEL_NAMES"
        return v

    def _get_contour_type(self, tag_name: str) -> Optional[str]:
        """
        return tag type corresponding to input tag_name
        Args:
            tag_name: name of the tag

        Returns:
            corresponding tag type
        """
        assert tag_name in TAG_MODEL_NAMES, f"tag_name is {tag_name}, which is not in TAG_MODEL_NAMES"
        tag_types = list(self.__fields__.keys())
        for i in tag_types:
            if tag_name in getattr(self, i):
                return i
        return None

    def _get_has_contour(self, tag_name: str) -> bool:
        assert tag_name in TAG_MODEL_NAMES, f"tag_name is {tag_name}, which not in TAG_MODEL_NAMES"
        if tag_name in self.no_contour:
            return False
        return True
