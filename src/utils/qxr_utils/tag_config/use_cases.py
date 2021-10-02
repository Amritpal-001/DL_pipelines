from enum import Enum
from typing import Dict, List, Optional

from qxr_utils.tag_config.constants import (
    ABNORMALITY_PREDICTION_MODELS,
    TAG_MODEL_NAMES,
    AbnormalThresholds,
    ConstEnum,
    ContoursToCombine,
    GeneralKeys,
    NormalThresholds,
    PatchThresholds,
    PixelThresholds,
    ScanThresholds,
)


class BaseUseCase(Enum):
    """
    Base use case class which inherits enum, has additional modifiers for thresholds
    """

    @classmethod
    def as_dict(cls) -> Dict:
        """Returns values defined in init part of enum as dict

        Returns:
                Dict: all the key, value pairs defined in the init part of enum
        """
        out = {k: v.value for k, v in cls.__members__.items()}
        return out

    @classmethod
    def modify_scan_thresholds(cls, scan_thresholds: Dict[str, float]) -> Optional[Dict[str, float]]:
        """function to apply any modifications to scan thresholds

        Args:
                scan_thresholds (Dict[str, float]): scan thresholds in the form of a dict

        Returns:
                Optional[Dict[str, float]]: modified scan thresholds
        """
        pass

    @classmethod
    def modify_patch_thresholds(cls, patch_thresholds: Dict[str, List[float]]) -> Optional[Dict[str, List[float]]]:
        """function to apply any modifications to patch thresholds

        Args:
                patch_thresholds (Dict[str, List[float]]): patch thresholds in the form of a dict

        Returns:
                Optional[Dict[str, List[float]]]: modified patch thresholds
        """
        pass

    @classmethod
    def modify_pixel_thresholds(cls, pixel_thresholds: Dict[str, float]) -> Optional[Dict[str, float]]:
        """function to apply any modifications to pixel thresholds

        Args:
                pixel_thresholds (Dict[str, float]): pixel thresholds in the form of a dict

        Returns:
                Optional[Dict[str, float]]: modified pixel thresholds
        """
        pass

    @classmethod
    def modify_normal_thresholds(cls, normal_thresholds: Dict[str, float]) -> Optional[Dict[str, float]]:
        """function to apply any modifications to normal thresholds

        Args:
                normal_thresholds (Dict[str, float]): normal thresholds in the form of a dict

        Returns:
                Optional[Dict[str, float]]: modified normal thresholds
        """
        pass

    @classmethod
    def modify_abnormal_thresholds(cls, abnormal_thresholds: Dict[str, float]) -> Optional[Dict[str, float]]:
        """function to apply any modifications to abnormal thresholds

        Args:
                abnormal_thresholds (Dict[str, float]): abnormal thresholds in the form of a dict

        Returns:
                Optional[Dict[str, float]]:  modified abnormal thresholds
        """
        pass

    @classmethod
    def get_use_case_dict(cls):
        """use this function to apply any modifications to thresholds and return the modified usecase as a dict.
        The format should like this

        out_dict = cls.as_dict() # get the defined values as a dict
        out_dict["pixel_thresholds"] = cls.modify_pixel_thresholds(out_dict["pixel_thresholds"]) # apply pixel threshold modifications
        out_dict["scan_thresholds"] = cls.modify_scan_thresholds(out_dict["scan_thresholds"]) # apply scan threshold modifications
        .
        .
        .
        return out_dict
        Raises:
                NotImplementedError
        """
        raise NotImplementedError


class nva_with_tbr_ce(BaseUseCase):
    tags_to_show = [
        i.value
        for i in [
            GeneralKeys.breathingtube,
            GeneralKeys.gastrictube,
            GeneralKeys.bluntedcp,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.nodule,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.ribfracture,
            GeneralKeys.opacity,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.tuberculosis,
            GeneralKeys.covid,
        ]
    ]
    tags_to_run = [
        i.value
        for i in [
            GeneralKeys.atelectasis,
            GeneralKeys.bluntedcp,
            GeneralKeys.calcification,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.diaphragm,
            GeneralKeys.degenspine,
            GeneralKeys.emphysema,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.mediastinalwidening,
            GeneralKeys.nodule,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.ribfracture,
            GeneralKeys.opacity,
            GeneralKeys.consolidation,
            GeneralKeys.cavity,
            GeneralKeys.fibrosis,
            GeneralKeys.nodule,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.scoliosis,
            GeneralKeys.trachealshift,
            GeneralKeys.tuberculosis,
        ]
    ]
    tags_to_combine = ContoursToCombine.as_dict()

    scan_thresholds = ScanThresholds.as_dict()
    patch_thresholds = PatchThresholds.as_dict()
    pixel_thresholds = PixelThresholds.as_dict()
    normal_thresholds = NormalThresholds.as_dict()
    abnormal_thresholds = AbnormalThresholds.as_dict()

    @classmethod
    def get_use_case_dict(cls):
        out_dict = cls.as_dict()
        return out_dict


class nva_without_tbr_ce(BaseUseCase):
    tags_to_show = [
        i.value
        for i in [
            GeneralKeys.breathingtube,
            GeneralKeys.gastrictube,
            GeneralKeys.bluntedcp,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.nodule,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.ribfracture,
            GeneralKeys.opacity,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.tuberculosis,
            GeneralKeys.covid,
        ]
    ]
    tags_to_run = [
        i.value
        for i in [
            GeneralKeys.atelectasis,
            GeneralKeys.bluntedcp,
            GeneralKeys.calcification,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.diaphragm,
            GeneralKeys.degenspine,
            GeneralKeys.emphysema,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.mediastinalwidening,
            GeneralKeys.nodule,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.ribfracture,
            GeneralKeys.opacity,
            GeneralKeys.consolidation,
            GeneralKeys.cavity,
            GeneralKeys.fibrosis,
            GeneralKeys.nodule,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.scoliosis,
            GeneralKeys.trachealshift,
            GeneralKeys.tuberculosis,
        ]
    ]
    tags_to_combine = ContoursToCombine.as_dict()

    scan_thresholds = ScanThresholds.as_dict()
    patch_thresholds = PatchThresholds.as_dict()
    pixel_thresholds = PixelThresholds.as_dict()
    normal_thresholds = NormalThresholds.as_dict()
    abnormal_thresholds = AbnormalThresholds.as_dict()

    @classmethod
    def modify_scan_thresholds(cls, scan_thresholds):
        scan_thresholds[GeneralKeys.atelectasis.value] = 0.9
        scan_thresholds[GeneralKeys.bluntedcp.value] = 0.7
        scan_thresholds[GeneralKeys.calcification.value] = 0.9
        scan_thresholds[GeneralKeys.cardiomegaly.value] = 0.75
        scan_thresholds[GeneralKeys.consolidation.value] = 0.35
        scan_thresholds[GeneralKeys.diaphragm.value] = 0.7
        scan_thresholds[GeneralKeys.emphysema.value] = 0.8
        scan_thresholds[GeneralKeys.fibrosis.value] = 0.7
        scan_thresholds[GeneralKeys.hilarlymphadenopathy.value] = 0.7
        scan_thresholds[GeneralKeys.nodule.value] = 0.45
        scan_thresholds[GeneralKeys.ribfracture.value] = 0.75
        scan_thresholds[GeneralKeys.opacity.value] = 0.3
        scan_thresholds[GeneralKeys.pleuraleffusion.value] = 0.6
        scan_thresholds[GeneralKeys.scoliosis.value] = 0.9

    @classmethod
    def modify_patch_thresholds(cls, patch_thresholds):
        patch_thresholds[GeneralKeys.opacity.value] = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        patch_thresholds[GeneralKeys.pleuraleffusion.value] = [1.0, 1.0, 0.6, 1.0, 1.0, 0.6]

    @classmethod
    def modify_pixel_thresholds(cls, pixel_thresholds):
        pixel_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.5

    @classmethod
    def modify_normal_thresholds(cls, normal_thresholds):
        normal_thresholds[GeneralKeys.atelectasis.value] = 0.9
        normal_thresholds[GeneralKeys.calcification.value] = 0.9
        normal_thresholds[GeneralKeys.cardiomegaly.value] = 0.75
        normal_thresholds[GeneralKeys.cavity.value] = 0.8
        normal_thresholds[GeneralKeys.consolidation.value] = 0.35
        normal_thresholds[GeneralKeys.degenspine.value] = 0.99
        normal_thresholds[GeneralKeys.emphysema.value] = 0.8
        normal_thresholds[GeneralKeys.mediastinalwidening.value] = 0.9
        normal_thresholds[GeneralKeys.nodule.value] = 0.45
        normal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        normal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.6
        normal_thresholds[GeneralKeys.pneumothorax.value] = 0.9
        normal_thresholds[GeneralKeys.trachealshift.value] = 0.9

    @classmethod
    def modify_abnormal_thresholds(cls, abnormal_thresholds):
        abnormal_thresholds[GeneralKeys.atelectasis.value] = 0.9
        abnormal_thresholds[GeneralKeys.bluntedcp.value] = 0.7
        abnormal_thresholds[GeneralKeys.calcification.value] = 0.9
        abnormal_thresholds[GeneralKeys.cardiomegaly.value] = 0.75
        abnormal_thresholds[GeneralKeys.cavity.value] = 0.8
        abnormal_thresholds[GeneralKeys.consolidation.value] = 0.35
        abnormal_thresholds[GeneralKeys.diaphragm.value] = 0.7
        abnormal_thresholds[GeneralKeys.degenspine.value] = 0.99
        abnormal_thresholds[GeneralKeys.emphysema.value] = 0.8
        abnormal_thresholds[GeneralKeys.fibrosis.value] = 0.7
        abnormal_thresholds[GeneralKeys.hilarlymphadenopathy.value] = 0.7
        abnormal_thresholds[GeneralKeys.nodule.value] = 0.45
        abnormal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        abnormal_thresholds[GeneralKeys.ribfracture.value] = 0.75
        abnormal_thresholds[GeneralKeys.opacity.value] = 0.3
        abnormal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.6
        abnormal_thresholds[GeneralKeys.pneumothorax.value] = 0.9
        abnormal_thresholds[GeneralKeys.scoliosis.value] = 0.9
        abnormal_thresholds[GeneralKeys.trachealshift.value] = 0.9
        abnormal_thresholds[GeneralKeys.tuberculosis.value] = 0.5

    @classmethod
    def get_use_case_dict(cls):
        out_dict = cls.as_dict()
        out_dict["scan_thresholds"] = cls.modify_pixel_thresholds(out_dict["scan_thresholds"])
        out_dict["patch_thresholds"] = cls.modify_pixel_thresholds(out_dict["patch_thresholds"])
        out_dict["pixel_thresholds"] = cls.modify_pixel_thresholds(out_dict["pixel_thresholds"])
        out_dict["normal_thresholds"] = cls.modify_pixel_thresholds(out_dict["normal_thresholds"])
        out_dict["abnormal_thresholds"] = cls.modify_pixel_thresholds(out_dict["abnormal_thresholds"])
        return out_dict


class nva_with_tbr_dev(BaseUseCase):
    tags_to_show = [
        i.value
        for i in [
            GeneralKeys.breathingtube,
            GeneralKeys.gastrictube,
            GeneralKeys.atelectasis,
            GeneralKeys.bluntedcp,
            GeneralKeys.calcification,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.diaphragm,
            GeneralKeys.emphysema,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.lung_nodule_malignancy,
            GeneralKeys.mediastinalwidening,
            GeneralKeys.nodule,
            GeneralKeys.pacemaker,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.ribfracture,
            GeneralKeys.opacity,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.scoliosis,
            GeneralKeys.sutures,
            GeneralKeys.trachealshift,
            GeneralKeys.tuberculosis,
            GeneralKeys.covid,
        ]
    ]
    tags_to_run = [
        i.value
        for i in [
            GeneralKeys.atelectasis,
            GeneralKeys.bluntedcp,
            GeneralKeys.calcification,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.diaphragm,
            GeneralKeys.degenspine,
            GeneralKeys.emphysema,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.lung_nodule_malignancy,
            GeneralKeys.mediastinalwidening,
            GeneralKeys.nodule,
            GeneralKeys.pacemaker,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.ribfracture,
            GeneralKeys.opacity,
            GeneralKeys.consolidation,
            GeneralKeys.cavity,
            GeneralKeys.fibrosis,
            GeneralKeys.nodule,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.scoliosis,
            GeneralKeys.sutures,
            GeneralKeys.trachealshift,
            GeneralKeys.tuberculosis,
        ]
    ]
    tags_to_combine = ContoursToCombine.as_dict()

    scan_thresholds = ScanThresholds.as_dict()
    patch_thresholds = PatchThresholds.as_dict()
    pixel_thresholds = PixelThresholds.as_dict()
    normal_thresholds = NormalThresholds.as_dict()
    abnormal_thresholds = AbnormalThresholds.as_dict()

    @classmethod
    def get_use_case_dict(cls):
        out_dict = cls.as_dict()
        return out_dict


class nva_without_tbr_dev(BaseUseCase):
    tags_to_show = [
        i.value
        for i in [
            GeneralKeys.breathingtube,
            GeneralKeys.gastrictube,
            GeneralKeys.atelectasis,
            GeneralKeys.bluntedcp,
            GeneralKeys.calcification,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.diaphragm,
            GeneralKeys.emphysema,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.mediastinalwidening,
            GeneralKeys.nodule,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.ribfracture,
            GeneralKeys.opacity,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.scoliosis,
            GeneralKeys.trachealshift,
            GeneralKeys.tuberculosis,
            GeneralKeys.covid,
        ]
    ]
    tags_to_run = [
        i.value
        for i in [
            GeneralKeys.atelectasis,
            GeneralKeys.bluntedcp,
            GeneralKeys.calcification,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.diaphragm,
            GeneralKeys.degenspine,
            GeneralKeys.emphysema,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.mediastinalwidening,
            GeneralKeys.nodule,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.ribfracture,
            GeneralKeys.opacity,
            GeneralKeys.consolidation,
            GeneralKeys.cavity,
            GeneralKeys.fibrosis,
            GeneralKeys.nodule,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.scoliosis,
            GeneralKeys.trachealshift,
            GeneralKeys.tuberculosis,
        ]
    ]
    tags_to_combine = ContoursToCombine.as_dict()

    scan_thresholds = ScanThresholds.as_dict()
    patch_thresholds = PatchThresholds.as_dict()
    pixel_thresholds = PixelThresholds.as_dict()
    normal_thresholds = NormalThresholds.as_dict()
    abnormal_thresholds = AbnormalThresholds.as_dict()

    @classmethod
    def modify_scan_thresholds(cls, scan_thresholds):
        scan_thresholds[GeneralKeys.atelectasis.value] = 0.9
        scan_thresholds[GeneralKeys.bluntedcp.value] = 0.7
        scan_thresholds[GeneralKeys.calcification.value] = 0.9
        scan_thresholds[GeneralKeys.cardiomegaly.value] = 0.75
        scan_thresholds[GeneralKeys.consolidation.value] = 0.35
        scan_thresholds[GeneralKeys.diaphragm.value] = 0.7
        scan_thresholds[GeneralKeys.emphysema.value] = 0.8
        scan_thresholds[GeneralKeys.fibrosis.value] = 0.7
        scan_thresholds[GeneralKeys.hilarlymphadenopathy.value] = 0.7
        scan_thresholds[GeneralKeys.nodule.value] = 0.45
        scan_thresholds[GeneralKeys.ribfracture.value] = 0.75
        scan_thresholds[GeneralKeys.opacity.value] = 0.3
        scan_thresholds[GeneralKeys.pleuraleffusion.value] = 0.6
        scan_thresholds[GeneralKeys.scoliosis.value] = 0.9

    @classmethod
    def modify_patch_thresholds(cls, patch_thresholds):
        patch_thresholds[GeneralKeys.opacity.value] = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        patch_thresholds[GeneralKeys.pleuraleffusion.value] = [1.0, 1.0, 0.6, 1.0, 1.0, 0.6]

    @classmethod
    def modify_pixel_thresholds(cls, pixel_thresholds):
        pixel_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.5

    @classmethod
    def modify_normal_thresholds(cls, normal_thresholds):
        normal_thresholds[GeneralKeys.atelectasis.value] = 0.9
        normal_thresholds[GeneralKeys.calcification.value] = 0.9
        normal_thresholds[GeneralKeys.cardiomegaly.value] = 0.75
        normal_thresholds[GeneralKeys.cavity.value] = 0.8
        normal_thresholds[GeneralKeys.consolidation.value] = 0.35
        normal_thresholds[GeneralKeys.degenspine.value] = 0.99
        normal_thresholds[GeneralKeys.emphysema.value] = 0.8
        normal_thresholds[GeneralKeys.mediastinalwidening.value] = 0.9
        normal_thresholds[GeneralKeys.nodule.value] = 0.45
        normal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        normal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.6
        normal_thresholds[GeneralKeys.pneumothorax.value] = 0.9
        normal_thresholds[GeneralKeys.trachealshift.value] = 0.9

    @classmethod
    def modify_abnormal_thresholds(cls, abnormal_thresholds):
        abnormal_thresholds[GeneralKeys.atelectasis.value] = 0.9
        abnormal_thresholds[GeneralKeys.bluntedcp.value] = 0.7
        abnormal_thresholds[GeneralKeys.calcification.value] = 0.9
        abnormal_thresholds[GeneralKeys.cardiomegaly.value] = 0.75
        abnormal_thresholds[GeneralKeys.cavity.value] = 0.8
        abnormal_thresholds[GeneralKeys.consolidation.value] = 0.35
        abnormal_thresholds[GeneralKeys.diaphragm.value] = 0.7
        abnormal_thresholds[GeneralKeys.degenspine.value] = 0.99
        abnormal_thresholds[GeneralKeys.emphysema.value] = 0.8
        abnormal_thresholds[GeneralKeys.fibrosis.value] = 0.7
        abnormal_thresholds[GeneralKeys.hilarlymphadenopathy.value] = 0.7
        abnormal_thresholds[GeneralKeys.nodule.value] = 0.45
        abnormal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        abnormal_thresholds[GeneralKeys.ribfracture.value] = 0.75
        abnormal_thresholds[GeneralKeys.opacity.value] = 0.3
        abnormal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.6
        abnormal_thresholds[GeneralKeys.pneumothorax.value] = 0.9
        abnormal_thresholds[GeneralKeys.scoliosis.value] = 0.9
        abnormal_thresholds[GeneralKeys.trachealshift.value] = 0.9
        abnormal_thresholds[GeneralKeys.tuberculosis.value] = 0.5

    @classmethod
    def get_use_case_dict(cls):
        out_dict = cls.as_dict()
        out_dict["scan_thresholds"] = cls.modify_pixel_thresholds(out_dict["scan_thresholds"])
        out_dict["patch_thresholds"] = cls.modify_pixel_thresholds(out_dict["patch_thresholds"])
        out_dict["pixel_thresholds"] = cls.modify_pixel_thresholds(out_dict["pixel_thresholds"])
        out_dict["normal_thresholds"] = cls.modify_pixel_thresholds(out_dict["normal_thresholds"])
        out_dict["abnormal_thresholds"] = cls.modify_pixel_thresholds(out_dict["abnormal_thresholds"])
        return out_dict


class onlylinesandtubes(BaseUseCase):
    tags_to_show = [
        i.value
        for i in [
            GeneralKeys.breathingtube,
            GeneralKeys.gastrictube,
        ]
    ]
    # TODO highlight that tags to run will be empty
    tags_to_run = []  # type: List[str]
    tags_to_combine = ContoursToCombine.as_dict()

    scan_thresholds = ScanThresholds.as_dict()
    patch_thresholds = PatchThresholds.as_dict()
    pixel_thresholds = PixelThresholds.as_dict()
    normal_thresholds = NormalThresholds.as_dict()
    abnormal_thresholds = AbnormalThresholds.as_dict()

    @classmethod
    def get_use_case_dict(cls):
        out_dict = cls.as_dict()
        return out_dict


class tb_screening(BaseUseCase):
    tags_to_show = [
        i.value
        for i in [
            GeneralKeys.bluntedcp,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.nodule,
            GeneralKeys.opacity,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.tuberculosis,
        ]
    ]
    tags_to_run = [
        i.value
        for i in [
            GeneralKeys.atelectasis,
            GeneralKeys.bluntedcp,
            GeneralKeys.calcification,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.nodule,
            GeneralKeys.opacity,
            GeneralKeys.consolidation,
            GeneralKeys.cavity,
            GeneralKeys.fibrosis,
            GeneralKeys.nodule,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.tuberculosis,
        ]
    ]
    tags_to_combine = ContoursToCombine.as_dict()

    scan_thresholds = ScanThresholds.as_dict()
    patch_thresholds = PatchThresholds.as_dict()
    pixel_thresholds = PixelThresholds.as_dict()
    normal_thresholds = NormalThresholds.as_dict()
    abnormal_thresholds = AbnormalThresholds.as_dict()

    @classmethod
    def modify_scan_thresholds(cls, scan_thresholds):
        scan_thresholds[GeneralKeys.atelectasis.value] = 0.9
        scan_thresholds[GeneralKeys.bluntedcp.value] = 0.85
        scan_thresholds[GeneralKeys.cardiomegaly.value] = 0.85
        scan_thresholds[GeneralKeys.cavity.value] = 0.9
        scan_thresholds[GeneralKeys.fibrosis.value] = 0.7
        scan_thresholds[GeneralKeys.nodule.value] = 0.5
        scan_thresholds[GeneralKeys.pleuraleffusion.value] = 0.75

    @classmethod
    def modify_patch_thresholds(cls, patch_thresholds):
        patch_thresholds[GeneralKeys.bluntedcp.value] = [1.0, 1.0, 0.85, 1.0, 1.0, 0.85]
        patch_thresholds[GeneralKeys.fibrosis.value] = [0.65, 0.65, 0.65, 0.65, 0.65, 0.65]
        patch_thresholds[GeneralKeys.nodule.value] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    @classmethod
    def modify_pixel_thresholds(cls, pixel_thresholds):
        pixel_thresholds[GeneralKeys.consolidation.value] = 0.4
        pixel_thresholds[GeneralKeys.fibrosis.value] = 0.5
        pixel_thresholds[GeneralKeys.nodule.value] = 0.4
        pixel_thresholds[GeneralKeys.opacity.value] = 0.4

    @classmethod
    def modify_normal_thresholds(cls, normal_thresholds):
        normal_thresholds[GeneralKeys.atelectasis.value] = 0.9
        normal_thresholds[GeneralKeys.bluntedcp.value] = 0.85
        normal_thresholds[GeneralKeys.calcification.value] = 0.85
        normal_thresholds[GeneralKeys.cardiomegaly.value] = 0.85
        normal_thresholds[GeneralKeys.cavity.value] = 0.9
        normal_thresholds[GeneralKeys.consolidation.value] = 0.5
        normal_thresholds[GeneralKeys.hilarlymphadenopathy.value] = 0.85
        normal_thresholds[GeneralKeys.opacity.value] = 0.5
        normal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        normal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.75

    @classmethod
    def modify_abnormal_thresholds(cls, abnormal_thresholds):
        abnormal_thresholds[GeneralKeys.atelectasis.value] = 0.9
        abnormal_thresholds[GeneralKeys.bluntedcp.value] = 0.85
        abnormal_thresholds[GeneralKeys.calcification.value] = 0.85
        abnormal_thresholds[GeneralKeys.cavity.value] = 0.9
        abnormal_thresholds[GeneralKeys.consolidation.value] = 0.5
        abnormal_thresholds[GeneralKeys.fibrosis.value] = 0.7
        abnormal_thresholds[GeneralKeys.hilarlymphadenopathy.value] = 0.85
        abnormal_thresholds[GeneralKeys.nodule.value] = 0.5
        abnormal_thresholds[GeneralKeys.opacity.value] = 0.5
        abnormal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        abnormal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.75
        abnormal_thresholds[GeneralKeys.tuberculosis.value] = 0.5

    @classmethod
    def get_use_case_dict(cls):
        out_dict = cls.as_dict()
        out_dict["scan_thresholds"] = cls.modify_pixel_thresholds(out_dict["scan_thresholds"])
        out_dict["patch_thresholds"] = cls.modify_pixel_thresholds(out_dict["patch_thresholds"])
        out_dict["pixel_thresholds"] = cls.modify_pixel_thresholds(out_dict["pixel_thresholds"])
        out_dict["normal_thresholds"] = cls.modify_pixel_thresholds(out_dict["normal_thresholds"])
        out_dict["abnormal_thresholds"] = cls.modify_pixel_thresholds(out_dict["abnormal_thresholds"])
        return out_dict


class scan_portal(BaseUseCase):
    tags_to_show = [
        i.value
        for i in [
            GeneralKeys.breathingtube,
            GeneralKeys.gastrictube,
            GeneralKeys.atelectasis,
            GeneralKeys.bluntedcp,
            GeneralKeys.calcification,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.diaphragm,
            GeneralKeys.emphysema,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.lung_nodule_malignancy,
            GeneralKeys.mediastinalwidening,
            GeneralKeys.nodule,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.ribfracture,
            GeneralKeys.opacity,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.scoliosis,
            GeneralKeys.trachealshift,
            GeneralKeys.tuberculosis,
        ]
    ]
    tags_to_run = [
        i.value
        for i in [
            GeneralKeys.atelectasis,
            GeneralKeys.bluntedcp,
            GeneralKeys.calcification,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.diaphragm,
            GeneralKeys.degenspine,
            GeneralKeys.emphysema,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.lung_nodule_malignancy,
            GeneralKeys.mediastinalwidening,
            GeneralKeys.nodule,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.ribfracture,
            GeneralKeys.opacity,
            GeneralKeys.consolidation,
            GeneralKeys.cavity,
            GeneralKeys.fibrosis,
            GeneralKeys.nodule,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.scoliosis,
            GeneralKeys.trachealshift,
            GeneralKeys.tuberculosis,
        ]
    ]
    tags_to_combine = ContoursToCombine.as_dict()

    scan_thresholds = ScanThresholds.as_dict()
    patch_thresholds = PatchThresholds.as_dict()
    pixel_thresholds = PixelThresholds.as_dict()
    normal_thresholds = NormalThresholds.as_dict()
    abnormal_thresholds = AbnormalThresholds.as_dict()

    @classmethod
    def modify_scan_thresholds(cls, scan_thresholds):
        scan_thresholds[GeneralKeys.atelectasis.value] = 0.9
        scan_thresholds[GeneralKeys.bluntedcp.value] = 0.85
        scan_thresholds[GeneralKeys.cavity.value] = 0.9
        scan_thresholds[GeneralKeys.emphysema.value] = 0.85
        scan_thresholds[GeneralKeys.fibrosis.value] = 0.7
        scan_thresholds[GeneralKeys.lung_nodule_malignancy.value] = 0.2
        scan_thresholds[GeneralKeys.nodule.value] = 0.5
        scan_thresholds[GeneralKeys.pleuraleffusion.value] = 0.75

    @classmethod
    def modify_patch_thresholds(cls, patch_thresholds):
        patch_thresholds[GeneralKeys.bluntedcp.value] = [1.0, 1.0, 0.85, 1.0, 1.0, 0.85]
        patch_thresholds[GeneralKeys.fibrosis.value] = [0.65, 0.65, 0.65, 0.65, 0.65, 0.65]
        patch_thresholds[GeneralKeys.nodule.value] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        patch_thresholds[GeneralKeys.pneumothorax.value] = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75]

    @classmethod
    def modify_pixel_thresholds(cls, pixel_thresholds):
        pixel_thresholds[GeneralKeys.bluntedcp.value] = 0.6
        pixel_thresholds[GeneralKeys.consolidation.value] = 0.4
        pixel_thresholds[GeneralKeys.fibrosis.value] = 0.5
        pixel_thresholds[GeneralKeys.nodule.value] = 0.4
        pixel_thresholds[GeneralKeys.opacity.value] = 0.4
        pixel_thresholds[GeneralKeys.pneumothorax.value] = 0.4

    @classmethod
    def modify_normal_thresholds(cls, normal_thresholds):
        normal_thresholds[GeneralKeys.atelectasis.value] = 0.9
        normal_thresholds[GeneralKeys.bluntedcp.value] = 0.85
        normal_thresholds[GeneralKeys.calcification.value] = 0.85
        normal_thresholds[GeneralKeys.cardiomegaly.value] = 0.8
        normal_thresholds[GeneralKeys.cavity.value] = 0.9
        normal_thresholds[GeneralKeys.consolidation.value] = 0.5
        normal_thresholds[GeneralKeys.diaphragm.value] = 0.75
        normal_thresholds[GeneralKeys.degenspine.value] = 0.99
        normal_thresholds[GeneralKeys.emphysema.value] = 0.85
        normal_thresholds[GeneralKeys.hilarlymphadenopathy.value] = 0.85
        normal_thresholds[GeneralKeys.lung_nodule_malignancy.value] = 0.2
        normal_thresholds[GeneralKeys.mediastinalwidening.value] = 0.9
        normal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        normal_thresholds[GeneralKeys.ribfracture.value] = 0.9
        normal_thresholds[GeneralKeys.opacity.value] = 0.5
        normal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.75
        normal_thresholds[GeneralKeys.pneumothorax.value] = 0.9
        normal_thresholds[GeneralKeys.scoliosis.value] = 0.95
        normal_thresholds[GeneralKeys.trachealshift.value] = 0.9

    @classmethod
    def modify_abnormal_thresholds(cls, abnormal_thresholds):
        abnormal_thresholds[GeneralKeys.atelectasis.value] = 0.9
        abnormal_thresholds[GeneralKeys.bluntedcp.value] = 0.85
        abnormal_thresholds[GeneralKeys.calcification.value] = 0.85
        abnormal_thresholds[GeneralKeys.cardiomegaly.value] = 0.8
        abnormal_thresholds[GeneralKeys.cavity.value] = 0.9
        abnormal_thresholds[GeneralKeys.consolidation.value] = 0.5
        abnormal_thresholds[GeneralKeys.diaphragm.value] = 0.75
        abnormal_thresholds[GeneralKeys.degenspine.value] = 0.99
        abnormal_thresholds[GeneralKeys.emphysema.value] = 0.85
        abnormal_thresholds[GeneralKeys.fibrosis.value] = 0.7
        abnormal_thresholds[GeneralKeys.hilarlymphadenopathy.value] = 0.85
        abnormal_thresholds[GeneralKeys.lung_nodule_malignancy.value] = 0.2
        abnormal_thresholds[GeneralKeys.nodule.value] = 0.5
        abnormal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        abnormal_thresholds[GeneralKeys.opacity.value] = 0.5
        abnormal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.75
        abnormal_thresholds[GeneralKeys.pneumothorax.value] = 0.9
        abnormal_thresholds[GeneralKeys.scoliosis.value] = 0.95
        abnormal_thresholds[GeneralKeys.trachealshift.value] = 0.9
        abnormal_thresholds[GeneralKeys.tuberculosis.value] = 0.5

    @classmethod
    def get_use_case_dict(cls):
        out_dict = cls.as_dict()
        out_dict["scan_thresholds"] = cls.modify_pixel_thresholds(out_dict["scan_thresholds"])
        out_dict["patch_thresholds"] = cls.modify_pixel_thresholds(out_dict["patch_thresholds"])
        out_dict["pixel_thresholds"] = cls.modify_pixel_thresholds(out_dict["pixel_thresholds"])
        out_dict["normal_thresholds"] = cls.modify_pixel_thresholds(out_dict["normal_thresholds"])
        out_dict["abnormal_thresholds"] = cls.modify_pixel_thresholds(out_dict["abnormal_thresholds"])
        return out_dict


class preread_ce(BaseUseCase):
    tags_to_show = [
        i.value
        for i in [
            GeneralKeys.breathingtube,
            GeneralKeys.gastrictube,
            GeneralKeys.bluntedcp,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.nodule,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.ribfracture,
            GeneralKeys.opacity,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.tuberculosis,
            GeneralKeys.covid,
        ]
    ]
    tags_to_run = [
        i.value
        for i in [
            GeneralKeys.atelectasis,
            GeneralKeys.bluntedcp,
            GeneralKeys.calcification,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.diaphragm,
            GeneralKeys.degenspine,
            GeneralKeys.emphysema,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.mediastinalwidening,
            GeneralKeys.nodule,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.ribfracture,
            GeneralKeys.opacity,
            GeneralKeys.consolidation,
            GeneralKeys.cavity,
            GeneralKeys.fibrosis,
            GeneralKeys.nodule,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.scoliosis,
            GeneralKeys.trachealshift,
            GeneralKeys.tuberculosis,
        ]
    ]
    tags_to_combine = ContoursToCombine.as_dict()

    scan_thresholds = ScanThresholds.as_dict()
    patch_thresholds = PatchThresholds.as_dict()
    pixel_thresholds = PixelThresholds.as_dict()
    normal_thresholds = NormalThresholds.as_dict()
    abnormal_thresholds = AbnormalThresholds.as_dict()

    @classmethod
    def modify_scan_thresholds(cls, scan_thresholds):
        scan_thresholds[GeneralKeys.atelectasis.value] = 0.95
        scan_thresholds[GeneralKeys.bluntedcp.value] = 0.8
        scan_thresholds[GeneralKeys.calcification.value] = 0.9
        scan_thresholds[GeneralKeys.cardiomegaly.value] = 0.85
        scan_thresholds[GeneralKeys.cavity.value] = 0.9
        scan_thresholds[GeneralKeys.consolidation.value] = 0.6
        scan_thresholds[GeneralKeys.diaphragm.value] = 0.8
        scan_thresholds[GeneralKeys.emphysema.value] = 0.8
        scan_thresholds[GeneralKeys.fibrosis.value] = 0.8
        scan_thresholds[GeneralKeys.hilarlymphadenopathy.value] = 0.9
        scan_thresholds[GeneralKeys.nodule.value] = 0.6
        scan_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.8
        scan_thresholds[GeneralKeys.ribfracture.value] = 0.78
        scan_thresholds[GeneralKeys.scoliosis.value] = 0.9
        scan_thresholds[GeneralKeys.trachealshift.value] = 0.95

    @classmethod
    def modify_patch_thresholds(cls, patch_thresholds):
        patch_thresholds[GeneralKeys.atelectasis.value] = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
        patch_thresholds[GeneralKeys.bluntedcp.value] = [1.0, 1.0, 0.8, 1.0, 1.0, 0.8]
        patch_thresholds[GeneralKeys.cavity.value] = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        patch_thresholds[GeneralKeys.pneumoperitoneum.value] = [1.0, 1.0, 0.8, 1.0, 1.0, 0.8]
        patch_thresholds[GeneralKeys.pneumothorax.value] = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

    @classmethod
    def modify_pixel_thresholds(cls, pixel_thresholds):
        pixel_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.5

    @classmethod
    def modify_normal_thresholds(cls, normal_thresholds):
        normal_thresholds[GeneralKeys.atelectasis.value] = 0.95
        normal_thresholds[GeneralKeys.bluntedcp.value] = 0.8
        normal_thresholds[GeneralKeys.calcification.value] = 0.9
        normal_thresholds[GeneralKeys.cardiomegaly.value] = 0.85
        normal_thresholds[GeneralKeys.cavity.value] = 0.9
        normal_thresholds[GeneralKeys.consolidation.value] = 0.6
        normal_thresholds[GeneralKeys.diaphragm.value] = 0.8
        normal_thresholds[GeneralKeys.degenspine.value] = 0.99
        normal_thresholds[GeneralKeys.emphysema.value] = 0.8
        normal_thresholds[GeneralKeys.fibrosis.value] = 0.8
        normal_thresholds[GeneralKeys.hilarlymphadenopathy.value] = 0.9
        normal_thresholds[GeneralKeys.mediastinalwidening.value] = 0.9
        normal_thresholds[GeneralKeys.nodule.value] = 0.6
        normal_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.8
        normal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        normal_thresholds[GeneralKeys.ribfracture.value] = 0.78
        normal_thresholds[GeneralKeys.opacity.value] = 0.5
        normal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.8
        normal_thresholds[GeneralKeys.pneumothorax.value] = 0.9
        normal_thresholds[GeneralKeys.trachealshift.value] = 0.95

    @classmethod
    def modify_abnormal_thresholds(cls, abnormal_thresholds):
        abnormal_thresholds[GeneralKeys.atelectasis.value] = 0.95
        abnormal_thresholds[GeneralKeys.bluntedcp.value] = 0.8
        abnormal_thresholds[GeneralKeys.calcification.value] = 0.9
        abnormal_thresholds[GeneralKeys.cavity.value] = 0.9
        abnormal_thresholds[GeneralKeys.consolidation.value] = 0.6
        abnormal_thresholds[GeneralKeys.degenspine.value] = 0.99
        abnormal_thresholds[GeneralKeys.emphysema.value] = 0.8
        abnormal_thresholds[GeneralKeys.fibrosis.value] = 0.8
        abnormal_thresholds[GeneralKeys.nodule.value] = 0.6
        abnormal_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.8
        abnormal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        abnormal_thresholds[GeneralKeys.ribfracture.value] = 0.78
        abnormal_thresholds[GeneralKeys.opacity.value] = 0.5
        abnormal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.8
        abnormal_thresholds[GeneralKeys.pneumothorax.value] = 0.9
        abnormal_thresholds[GeneralKeys.scoliosis.value] = 0.9
        abnormal_thresholds[GeneralKeys.trachealshift.value] = 0.95
        abnormal_thresholds[GeneralKeys.tuberculosis.value] = 0.5

    @classmethod
    def get_use_case_dict(cls):
        out_dict = cls.as_dict()
        out_dict["scan_thresholds"] = cls.modify_pixel_thresholds(out_dict["scan_thresholds"])
        out_dict["patch_thresholds"] = cls.modify_pixel_thresholds(out_dict["patch_thresholds"])
        out_dict["pixel_thresholds"] = cls.modify_pixel_thresholds(out_dict["pixel_thresholds"])
        out_dict["normal_thresholds"] = cls.modify_pixel_thresholds(out_dict["normal_thresholds"])
        out_dict["abnormal_thresholds"] = cls.modify_pixel_thresholds(out_dict["abnormal_thresholds"])
        return out_dict


class preread_dev(BaseUseCase):
    tags_to_show = [
        i.value
        for i in [
            GeneralKeys.breathingtube,
            GeneralKeys.gastrictube,
            GeneralKeys.bluntedcp,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.diaphragm,
            GeneralKeys.emphysema,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.mediastinalwidening,
            GeneralKeys.nodule,
            GeneralKeys.pacemaker,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.ribfracture,
            GeneralKeys.opacity,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.scoliosis,
            GeneralKeys.sutures,
            GeneralKeys.trachealshift,
            GeneralKeys.tuberculosis,
            GeneralKeys.covid,
        ]
    ]
    tags_to_run = [
        i.value
        for i in [
            GeneralKeys.atelectasis,
            GeneralKeys.bluntedcp,
            GeneralKeys.calcification,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.diaphragm,
            GeneralKeys.degenspine,
            GeneralKeys.emphysema,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.mediastinalwidening,
            GeneralKeys.nodule,
            GeneralKeys.pacemaker,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.ribfracture,
            GeneralKeys.opacity,
            GeneralKeys.consolidation,
            GeneralKeys.cavity,
            GeneralKeys.fibrosis,
            GeneralKeys.nodule,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.scoliosis,
            GeneralKeys.sutures,
            GeneralKeys.trachealshift,
            GeneralKeys.tuberculosis,
        ]
    ]
    tags_to_combine = ContoursToCombine.as_dict()

    scan_thresholds = ScanThresholds.as_dict()
    patch_thresholds = PatchThresholds.as_dict()
    pixel_thresholds = PixelThresholds.as_dict()
    normal_thresholds = NormalThresholds.as_dict()
    abnormal_thresholds = AbnormalThresholds.as_dict()

    @classmethod
    def modify_scan_thresholds(cls, scan_thresholds):
        scan_thresholds[GeneralKeys.atelectasis.value] = 0.95
        scan_thresholds[GeneralKeys.bluntedcp.value] = 0.8
        scan_thresholds[GeneralKeys.calcification.value] = 0.9
        scan_thresholds[GeneralKeys.cardiomegaly.value] = 0.85
        scan_thresholds[GeneralKeys.cavity.value] = 0.9
        scan_thresholds[GeneralKeys.consolidation.value] = 0.6
        scan_thresholds[GeneralKeys.diaphragm.value] = 0.8
        scan_thresholds[GeneralKeys.emphysema.value] = 0.8
        scan_thresholds[GeneralKeys.fibrosis.value] = 0.8
        scan_thresholds[GeneralKeys.hilarlymphadenopathy.value] = 0.9
        scan_thresholds[GeneralKeys.nodule.value] = 0.6
        scan_thresholds[GeneralKeys.pacemaker.value] = 0.7
        scan_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.8
        scan_thresholds[GeneralKeys.ribfracture.value] = 0.78
        scan_thresholds[GeneralKeys.scoliosis.value] = 0.9
        scan_thresholds[GeneralKeys.trachealshift.value] = 0.95

    @classmethod
    def modify_patch_thresholds(cls, patch_thresholds):
        patch_thresholds[GeneralKeys.atelectasis.value] = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
        patch_thresholds[GeneralKeys.bluntedcp.value] = [1.0, 1.0, 0.8, 1.0, 1.0, 0.8]
        patch_thresholds[GeneralKeys.cavity.value] = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        patch_thresholds[GeneralKeys.pneumoperitoneum.value] = [1.0, 1.0, 0.8, 1.0, 1.0, 0.8]
        patch_thresholds[GeneralKeys.pneumothorax.value] = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

    @classmethod
    def modify_pixel_thresholds(cls, pixel_thresholds):
        pixel_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.5

    @classmethod
    def modify_normal_thresholds(cls, normal_thresholds):
        normal_thresholds[GeneralKeys.atelectasis.value] = 0.95
        normal_thresholds[GeneralKeys.bluntedcp.value] = 0.8
        normal_thresholds[GeneralKeys.calcification.value] = 0.9
        normal_thresholds[GeneralKeys.cardiomegaly.value] = 0.85
        normal_thresholds[GeneralKeys.cavity.value] = 0.9
        normal_thresholds[GeneralKeys.consolidation.value] = 0.6
        normal_thresholds[GeneralKeys.diaphragm.value] = 0.8
        normal_thresholds[GeneralKeys.degenspine.value] = 0.99
        normal_thresholds[GeneralKeys.emphysema.value] = 0.8
        normal_thresholds[GeneralKeys.fibrosis.value] = 0.8
        normal_thresholds[GeneralKeys.hilarlymphadenopathy.value] = 0.9
        normal_thresholds[GeneralKeys.mediastinalwidening.value] = 0.9
        normal_thresholds[GeneralKeys.nodule.value] = 0.6
        normal_thresholds[GeneralKeys.pacemaker.value] = 0.7
        normal_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.8
        normal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        normal_thresholds[GeneralKeys.ribfracture.value] = 0.78
        normal_thresholds[GeneralKeys.opacity.value] = 0.5
        normal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.8
        normal_thresholds[GeneralKeys.pneumothorax.value] = 0.9
        normal_thresholds[GeneralKeys.trachealshift.value] = 0.95

    @classmethod
    def modify_abnormal_thresholds(cls, abnormal_thresholds):
        abnormal_thresholds[GeneralKeys.atelectasis.value] = 0.95
        abnormal_thresholds[GeneralKeys.bluntedcp.value] = 0.8
        abnormal_thresholds[GeneralKeys.calcification.value] = 0.9
        abnormal_thresholds[GeneralKeys.cavity.value] = 0.9
        abnormal_thresholds[GeneralKeys.consolidation.value] = 0.6
        abnormal_thresholds[GeneralKeys.degenspine.value] = 0.99
        abnormal_thresholds[GeneralKeys.emphysema.value] = 0.8
        abnormal_thresholds[GeneralKeys.fibrosis.value] = 0.8
        abnormal_thresholds[GeneralKeys.nodule.value] = 0.6
        abnormal_thresholds[GeneralKeys.pacemaker.value] = 0.7
        abnormal_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.8
        abnormal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        abnormal_thresholds[GeneralKeys.ribfracture.value] = 0.78
        abnormal_thresholds[GeneralKeys.opacity.value] = 0.5
        abnormal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.8
        abnormal_thresholds[GeneralKeys.pneumothorax.value] = 0.9
        abnormal_thresholds[GeneralKeys.scoliosis.value] = 0.9
        abnormal_thresholds[GeneralKeys.trachealshift.value] = 0.95
        abnormal_thresholds[GeneralKeys.tuberculosis.value] = 0.5

    @classmethod
    def get_use_case_dict(cls):
        out_dict = cls.as_dict()
        out_dict["scan_thresholds"] = cls.modify_pixel_thresholds(out_dict["scan_thresholds"])
        out_dict["patch_thresholds"] = cls.modify_pixel_thresholds(out_dict["patch_thresholds"])
        out_dict["pixel_thresholds"] = cls.modify_pixel_thresholds(out_dict["pixel_thresholds"])
        out_dict["normal_thresholds"] = cls.modify_pixel_thresholds(out_dict["normal_thresholds"])
        out_dict["abnormal_thresholds"] = cls.modify_pixel_thresholds(out_dict["abnormal_thresholds"])
        return out_dict


class preread_nolt_ce(BaseUseCase):
    tags_to_show = [
        i.value
        for i in [
            GeneralKeys.bluntedcp,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.nodule,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.ribfracture,
            GeneralKeys.opacity,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.tuberculosis,
            GeneralKeys.covid,
        ]
    ]
    tags_to_run = [
        i.value
        for i in [
            GeneralKeys.atelectasis,
            GeneralKeys.bluntedcp,
            GeneralKeys.calcification,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.diaphragm,
            GeneralKeys.degenspine,
            GeneralKeys.emphysema,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.mediastinalwidening,
            GeneralKeys.nodule,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.ribfracture,
            GeneralKeys.opacity,
            GeneralKeys.consolidation,
            GeneralKeys.cavity,
            GeneralKeys.fibrosis,
            GeneralKeys.nodule,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.scoliosis,
            GeneralKeys.trachealshift,
            GeneralKeys.tuberculosis,
        ]
    ]
    tags_to_combine = ContoursToCombine.as_dict()

    scan_thresholds = ScanThresholds.as_dict()
    patch_thresholds = PatchThresholds.as_dict()
    pixel_thresholds = PixelThresholds.as_dict()
    normal_thresholds = NormalThresholds.as_dict()
    abnormal_thresholds = AbnormalThresholds.as_dict()

    @classmethod
    def modify_scan_thresholds(cls, scan_thresholds):
        scan_thresholds[GeneralKeys.atelectasis.value] = 0.95
        scan_thresholds[GeneralKeys.bluntedcp.value] = 0.8
        scan_thresholds[GeneralKeys.calcification.value] = 0.9
        scan_thresholds[GeneralKeys.cardiomegaly.value] = 0.85
        scan_thresholds[GeneralKeys.cavity.value] = 0.9
        scan_thresholds[GeneralKeys.consolidation.value] = 0.6
        scan_thresholds[GeneralKeys.diaphragm.value] = 0.8
        scan_thresholds[GeneralKeys.emphysema.value] = 0.8
        scan_thresholds[GeneralKeys.fibrosis.value] = 0.8
        scan_thresholds[GeneralKeys.hilarlymphadenopathy.value] = 0.9
        scan_thresholds[GeneralKeys.nodule.value] = 0.6
        scan_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.8
        scan_thresholds[GeneralKeys.ribfracture.value] = 0.78
        scan_thresholds[GeneralKeys.scoliosis.value] = 0.9
        scan_thresholds[GeneralKeys.trachealshift.value] = 0.95

    @classmethod
    def modify_patch_thresholds(cls, patch_thresholds):
        patch_thresholds[GeneralKeys.atelectasis.value] = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
        patch_thresholds[GeneralKeys.bluntedcp.value] = [1.0, 1.0, 0.8, 1.0, 1.0, 0.8]
        patch_thresholds[GeneralKeys.cavity.value] = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        patch_thresholds[GeneralKeys.pneumoperitoneum.value] = [1.0, 1.0, 0.8, 1.0, 1.0, 0.8]
        patch_thresholds[GeneralKeys.pneumothorax.value] = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

    @classmethod
    def modify_pixel_thresholds(cls, pixel_thresholds):
        pixel_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.5

    @classmethod
    def modify_normal_thresholds(cls, normal_thresholds):
        normal_thresholds[GeneralKeys.atelectasis.value] = 0.95
        normal_thresholds[GeneralKeys.bluntedcp.value] = 0.8
        normal_thresholds[GeneralKeys.calcification.value] = 0.9
        normal_thresholds[GeneralKeys.cardiomegaly.value] = 0.85
        normal_thresholds[GeneralKeys.cavity.value] = 0.9
        normal_thresholds[GeneralKeys.consolidation.value] = 0.6
        normal_thresholds[GeneralKeys.diaphragm.value] = 0.8
        normal_thresholds[GeneralKeys.degenspine.value] = 0.99
        normal_thresholds[GeneralKeys.emphysema.value] = 0.8
        normal_thresholds[GeneralKeys.fibrosis.value] = 0.8
        normal_thresholds[GeneralKeys.hilarlymphadenopathy.value] = 0.9
        normal_thresholds[GeneralKeys.mediastinalwidening.value] = 0.9
        normal_thresholds[GeneralKeys.nodule.value] = 0.6
        normal_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.8
        normal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        normal_thresholds[GeneralKeys.ribfracture.value] = 0.78
        normal_thresholds[GeneralKeys.opacity.value] = 0.5
        normal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.8
        normal_thresholds[GeneralKeys.pneumothorax.value] = 0.9
        normal_thresholds[GeneralKeys.trachealshift.value] = 0.95

    @classmethod
    def modify_abnormal_thresholds(cls, abnormal_thresholds):
        abnormal_thresholds[GeneralKeys.atelectasis.value] = 0.95
        abnormal_thresholds[GeneralKeys.bluntedcp.value] = 0.8
        abnormal_thresholds[GeneralKeys.calcification.value] = 0.9
        abnormal_thresholds[GeneralKeys.cavity.value] = 0.9
        abnormal_thresholds[GeneralKeys.consolidation.value] = 0.6
        abnormal_thresholds[GeneralKeys.degenspine.value] = 0.99
        abnormal_thresholds[GeneralKeys.emphysema.value] = 0.8
        abnormal_thresholds[GeneralKeys.fibrosis.value] = 0.8
        abnormal_thresholds[GeneralKeys.nodule.value] = 0.6
        abnormal_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.8
        abnormal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        abnormal_thresholds[GeneralKeys.ribfracture.value] = 0.78
        abnormal_thresholds[GeneralKeys.opacity.value] = 0.5
        abnormal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.8
        abnormal_thresholds[GeneralKeys.pneumothorax.value] = 0.9
        abnormal_thresholds[GeneralKeys.scoliosis.value] = 0.9
        abnormal_thresholds[GeneralKeys.trachealshift.value] = 0.95
        abnormal_thresholds[GeneralKeys.tuberculosis.value] = 0.5

    @classmethod
    def get_use_case_dict(cls):
        out_dict = cls.as_dict()
        out_dict["scan_thresholds"] = cls.modify_pixel_thresholds(out_dict["scan_thresholds"])
        out_dict["patch_thresholds"] = cls.modify_pixel_thresholds(out_dict["patch_thresholds"])
        out_dict["pixel_thresholds"] = cls.modify_pixel_thresholds(out_dict["pixel_thresholds"])
        out_dict["normal_thresholds"] = cls.modify_pixel_thresholds(out_dict["normal_thresholds"])
        out_dict["abnormal_thresholds"] = cls.modify_pixel_thresholds(out_dict["abnormal_thresholds"])
        return out_dict


class covid_tb_only(BaseUseCase):
    tags_to_show = [
        i.value
        for i in [
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.opacity,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.tuberculosis,
            GeneralKeys.covid,
        ]
    ]
    tags_to_run = [
        i.value
        for i in [
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.opacity,
            GeneralKeys.consolidation,
            GeneralKeys.cavity,
            GeneralKeys.fibrosis,
            GeneralKeys.nodule,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.tuberculosis,
        ]
    ]
    tags_to_combine = ContoursToCombine.as_dict()

    scan_thresholds = ScanThresholds.as_dict()
    patch_thresholds = PatchThresholds.as_dict()
    pixel_thresholds = PixelThresholds.as_dict()
    normal_thresholds = NormalThresholds.as_dict()
    abnormal_thresholds = AbnormalThresholds.as_dict()

    @classmethod
    def modify_scan_thresholds(cls, scan_thresholds):
        scan_thresholds[GeneralKeys.cavity.value] = 0.9
        scan_thresholds[GeneralKeys.fibrosis.value] = 0.7
        scan_thresholds[GeneralKeys.nodule.value] = 0.5
        scan_thresholds[GeneralKeys.pleuraleffusion.value] = 0.75

    @classmethod
    def modify_patch_thresholds(cls, patch_thresholds):
        patch_thresholds[GeneralKeys.fibrosis.value] = [0.65, 0.65, 0.65, 0.65, 0.65, 0.65]
        patch_thresholds[GeneralKeys.nodule.value] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    @classmethod
    def modify_pixel_thresholds(cls, pixel_thresholds):
        pixel_thresholds[GeneralKeys.consolidation.value] = 0.4
        pixel_thresholds[GeneralKeys.opacity.value] = 0.4
        pixel_thresholds[GeneralKeys.fibrosis.value] = 0.5
        pixel_thresholds[GeneralKeys.nodule.value] = 0.4

    @classmethod
    def modify_normal_thresholds(cls, normal_thresholds):
        normal_thresholds[GeneralKeys.cavity.value] = 0.9
        normal_thresholds[GeneralKeys.consolidation.value] = 0.5
        normal_thresholds[GeneralKeys.opacity.value] = 0.5
        normal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        normal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.75

    @classmethod
    def modify_abnormal_thresholds(cls, abnormal_thresholds):
        abnormal_thresholds[GeneralKeys.cavity.value] = 0.9
        abnormal_thresholds[GeneralKeys.consolidation.value] = 0.5
        abnormal_thresholds[GeneralKeys.opacity.value] = 0.5
        abnormal_thresholds[GeneralKeys.fibrosis.value] = 0.7
        abnormal_thresholds[GeneralKeys.nodule.value] = 0.5
        abnormal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        abnormal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.75
        abnormal_thresholds[GeneralKeys.tuberculosis.value] = 0.5

    @classmethod
    def get_use_case_dict(cls):
        out_dict = cls.as_dict()
        out_dict["scan_thresholds"] = cls.modify_pixel_thresholds(out_dict["scan_thresholds"])
        out_dict["patch_thresholds"] = cls.modify_pixel_thresholds(out_dict["patch_thresholds"])
        out_dict["pixel_thresholds"] = cls.modify_pixel_thresholds(out_dict["pixel_thresholds"])
        out_dict["normal_thresholds"] = cls.modify_pixel_thresholds(out_dict["normal_thresholds"])
        out_dict["abnormal_thresholds"] = cls.modify_pixel_thresholds(out_dict["abnormal_thresholds"])
        return out_dict


class az_latam(BaseUseCase):
    tags_to_show = [
        i.value
        for i in [
            GeneralKeys.cardiomegaly,
            GeneralKeys.consolidation,
            GeneralKeys.emphysema,
            GeneralKeys.lung_nodule_malignancy,
            GeneralKeys.mediastinalwidening,
            GeneralKeys.nodule,
            GeneralKeys.opacity,
            GeneralKeys.tuberculosis,
            GeneralKeys.covid,
        ]
    ]
    tags_to_run = [
        i.value
        for i in [
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.emphysema,
            GeneralKeys.lung_nodule_malignancy,
            GeneralKeys.mediastinalwidening,
            GeneralKeys.nodule,
            GeneralKeys.opacity,
            GeneralKeys.consolidation,
            GeneralKeys.cavity,
            GeneralKeys.fibrosis,
            GeneralKeys.nodule,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.tuberculosis,
        ]
    ]
    tags_to_combine = ContoursToCombine.as_dict()

    scan_thresholds = ScanThresholds.as_dict()
    patch_thresholds = PatchThresholds.as_dict()
    pixel_thresholds = PixelThresholds.as_dict()
    normal_thresholds = NormalThresholds.as_dict()
    abnormal_thresholds = AbnormalThresholds.as_dict()

    @classmethod
    def modify_scan_thresholds(cls, scan_thresholds):
        scan_thresholds[GeneralKeys.cardiomegaly.value] = 0.85
        scan_thresholds[GeneralKeys.cavity.value] = 0.9
        scan_thresholds[GeneralKeys.nodule.value] = 0.4
        scan_thresholds[GeneralKeys.fibrosis.value] = 0.7
        scan_thresholds[GeneralKeys.pleuraleffusion.value] = 0.75

    @classmethod
    def modify_patch_thresholds(cls, patch_thresholds):
        patch_thresholds[GeneralKeys.nodule.value] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        patch_thresholds[GeneralKeys.fibrosis.value] = [0.65, 0.65, 0.65, 0.65, 0.65, 0.65]

    @classmethod
    def modify_pixel_thresholds(cls, pixel_thresholds):
        pixel_thresholds[GeneralKeys.consolidation.value] = 0.4
        pixel_thresholds[GeneralKeys.nodule.value] = 0.4
        pixel_thresholds[GeneralKeys.opacity.value] = 0.4
        pixel_thresholds[GeneralKeys.fibrosis.value] = 0.5

    @classmethod
    def modify_normal_thresholds(cls, normal_thresholds):
        normal_thresholds[GeneralKeys.cardiomegaly.value] = 0.85
        normal_thresholds[GeneralKeys.cavity.value] = 0.9
        normal_thresholds[GeneralKeys.consolidation.value] = 0.5
        normal_thresholds[GeneralKeys.emphysema.value] = 0.9
        normal_thresholds[GeneralKeys.mediastinalwidening.value] = 0.9
        normal_thresholds[GeneralKeys.nodule.value] = 0.4
        normal_thresholds[GeneralKeys.opacity.value] = 0.5
        normal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        normal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.75

    @classmethod
    def modify_abnormal_thresholds(cls, abnormal_thresholds):
        abnormal_thresholds[GeneralKeys.cavity.value] = 0.9
        abnormal_thresholds[GeneralKeys.consolidation.value] = 0.5
        abnormal_thresholds[GeneralKeys.nodule.value] = 0.4
        abnormal_thresholds[GeneralKeys.opacity.value] = 0.5
        abnormal_thresholds[GeneralKeys.fibrosis.value] = 0.7
        abnormal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        abnormal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.75
        abnormal_thresholds[GeneralKeys.tuberculosis.value] = 0.5

    @classmethod
    def get_use_case_dict(cls):
        out_dict = cls.as_dict()
        out_dict["scan_thresholds"] = cls.modify_pixel_thresholds(out_dict["scan_thresholds"])
        out_dict["patch_thresholds"] = cls.modify_pixel_thresholds(out_dict["patch_thresholds"])
        out_dict["pixel_thresholds"] = cls.modify_pixel_thresholds(out_dict["pixel_thresholds"])
        out_dict["normal_thresholds"] = cls.modify_pixel_thresholds(out_dict["normal_thresholds"])
        out_dict["abnormal_thresholds"] = cls.modify_pixel_thresholds(out_dict["abnormal_thresholds"])
        return out_dict


class preread_ce_c2(BaseUseCase):
    tags_to_show = [
        i.value
        for i in [
            GeneralKeys.bluntedcp,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.nodule,
            GeneralKeys.opacity,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.tuberculosis,
        ]
    ]
    tags_to_run = [
        i.value
        for i in [
            GeneralKeys.atelectasis,
            GeneralKeys.bluntedcp,
            GeneralKeys.calcification,
            GeneralKeys.cardiomegaly,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.diaphragm,
            GeneralKeys.degenspine,
            GeneralKeys.emphysema,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.mediastinalwidening,
            GeneralKeys.nodule,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.ribfracture,
            GeneralKeys.opacity,
            GeneralKeys.consolidation,
            GeneralKeys.cavity,
            GeneralKeys.fibrosis,
            GeneralKeys.nodule,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumothorax,
            GeneralKeys.scoliosis,
            GeneralKeys.trachealshift,
            GeneralKeys.tuberculosis,
        ]
    ]
    tags_to_combine = ContoursToCombine.as_dict()

    scan_thresholds = ScanThresholds.as_dict()
    patch_thresholds = PatchThresholds.as_dict()
    pixel_thresholds = PixelThresholds.as_dict()
    normal_thresholds = NormalThresholds.as_dict()
    abnormal_thresholds = AbnormalThresholds.as_dict()

    @classmethod
    def modify_scan_thresholds(cls, scan_thresholds):
        scan_thresholds[GeneralKeys.atelectasis.value] = 0.95
        scan_thresholds[GeneralKeys.bluntedcp.value] = 0.8
        scan_thresholds[GeneralKeys.calcification.value] = 0.9
        scan_thresholds[GeneralKeys.cardiomegaly.value] = 0.85
        scan_thresholds[GeneralKeys.cavity.value] = 0.9
        scan_thresholds[GeneralKeys.consolidation.value] = 0.6
        scan_thresholds[GeneralKeys.diaphragm.value] = 0.95
        scan_thresholds[GeneralKeys.fibrosis.value] = 0.8
        scan_thresholds[GeneralKeys.hilarlymphadenopathy.value] = 0.9
        scan_thresholds[GeneralKeys.nodule.value] = 0.6
        scan_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.8
        scan_thresholds[GeneralKeys.ribfracture.value] = 0.78
        scan_thresholds[GeneralKeys.scoliosis.value] = 0.9
        scan_thresholds[GeneralKeys.trachealshift.value] = 0.95

    @classmethod
    def modify_patch_thresholds(cls, patch_thresholds):
        patch_thresholds[GeneralKeys.atelectasis.value] = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
        patch_thresholds[GeneralKeys.bluntedcp.value] = [1.0, 1.0, 0.8, 1.0, 1.0, 0.8]
        patch_thresholds[GeneralKeys.cavity.value] = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        patch_thresholds[GeneralKeys.pneumoperitoneum.value] = [1.0, 1.0, 0.8, 1.0, 1.0, 0.8]
        patch_thresholds[GeneralKeys.pneumothorax.value] = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

    @classmethod
    def modify_pixel_thresholds(cls, pixel_thresholds):
        pixel_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.5

    @classmethod
    def modify_normal_thresholds(cls, normal_thresholds):
        normal_thresholds[GeneralKeys.atelectasis.value] = 0.95
        normal_thresholds[GeneralKeys.bluntedcp.value] = 0.8
        normal_thresholds[GeneralKeys.calcification.value] = 0.9
        normal_thresholds[GeneralKeys.cardiomegaly.value] = 0.85
        normal_thresholds[GeneralKeys.cavity.value] = 0.9
        normal_thresholds[GeneralKeys.consolidation.value] = 0.6
        normal_thresholds[GeneralKeys.diaphragm.value] = 0.95
        normal_thresholds[GeneralKeys.degenspine.value] = 0.99
        normal_thresholds[GeneralKeys.emphysema.value] = 0.9
        normal_thresholds[GeneralKeys.fibrosis.value] = 0.8
        normal_thresholds[GeneralKeys.hilarlymphadenopathy.value] = 0.9
        normal_thresholds[GeneralKeys.mediastinalwidening.value] = 0.9
        normal_thresholds[GeneralKeys.nodule.value] = 0.6
        normal_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.8
        normal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        normal_thresholds[GeneralKeys.ribfracture.value] = 0.78
        normal_thresholds[GeneralKeys.opacity.value] = 0.5
        normal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.8
        normal_thresholds[GeneralKeys.pneumothorax.value] = 0.9
        normal_thresholds[GeneralKeys.trachealshift.value] = 0.95

    @classmethod
    def modify_abnormal_thresholds(cls, abnormal_thresholds):
        abnormal_thresholds[GeneralKeys.atelectasis.value] = 0.95
        abnormal_thresholds[GeneralKeys.bluntedcp.value] = 0.8
        abnormal_thresholds[GeneralKeys.calcification.value] = 0.9
        abnormal_thresholds[GeneralKeys.cavity.value] = 0.9
        abnormal_thresholds[GeneralKeys.consolidation.value] = 0.6
        abnormal_thresholds[GeneralKeys.diaphragm.value] = 0.95
        abnormal_thresholds[GeneralKeys.degenspine.value] = 0.99
        abnormal_thresholds[GeneralKeys.fibrosis.value] = 0.8
        abnormal_thresholds[GeneralKeys.nodule.value] = 0.6
        abnormal_thresholds[GeneralKeys.pneumoperitoneum.value] = 0.8
        abnormal_thresholds[GeneralKeys.reticulonodularpattern.value] = 0.9
        abnormal_thresholds[GeneralKeys.ribfracture.value] = 0.78
        abnormal_thresholds[GeneralKeys.opacity.value] = 0.5
        abnormal_thresholds[GeneralKeys.pleuraleffusion.value] = 0.8
        abnormal_thresholds[GeneralKeys.pneumothorax.value] = 0.9
        abnormal_thresholds[GeneralKeys.scoliosis.value] = 0.9
        abnormal_thresholds[GeneralKeys.trachealshift.value] = 0.95
        abnormal_thresholds[GeneralKeys.tuberculosis.value] = 0.5

    @classmethod
    def get_use_case_dict(cls):
        out_dict = cls.as_dict()
        out_dict["scan_thresholds"] = cls.modify_pixel_thresholds(out_dict["scan_thresholds"])
        out_dict["patch_thresholds"] = cls.modify_pixel_thresholds(out_dict["patch_thresholds"])
        out_dict["pixel_thresholds"] = cls.modify_pixel_thresholds(out_dict["pixel_thresholds"])
        out_dict["normal_thresholds"] = cls.modify_pixel_thresholds(out_dict["normal_thresholds"])
        out_dict["abnormal_thresholds"] = cls.modify_pixel_thresholds(out_dict["abnormal_thresholds"])
        return out_dict
