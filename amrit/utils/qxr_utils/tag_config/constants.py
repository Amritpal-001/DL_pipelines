from enum import Enum, unique
from typing import Dict, List


class ConstEnum(Enum):
    """
    Baseclass which inherits enum, has 2 additional members as_dict and key_names
    So why enum instead of class? because it's easy to get all the members and values as a dict,
    while retaining the power of a class

    """

    @classmethod
    def as_dict(cls) -> Dict[str, List[str]]:
        """
        Returns: This enum as a dict

        """
        out = {k: v.value for k, v in cls.__members__.items()}
        return out

    @classmethod
    def key_names(cls) -> List[str]:
        """

        Returns: all the names of this enum

        """
        out = [i.name for i in cls]
        return out


@unique
class GeneralKeys(ConstEnum):
    heatmap = "heatmap"
    report = "report"
    patch_borders = "patch_borders"
    model_config = "model_config"
    ensemble_scan = "ensemble_scan"
    ensemble_patch = "ensemble_patch"
    ensemble_lung = "ensemble_lung"
    ensemble_pixel = "ensemble_pixel"
    threshold = "threshold"
    model_type = "model_type"
    scan_threshold = "scan_threshold"
    patch_threshold = "patch_threshold"
    pixel_threshold = "pixel_threshold"
    abnormal_threshold = "abnormal_threshold"
    normal_threshold = "normal_threshold"
    thresholded_score = "thresholded_score"
    score = "score"
    name = "name"
    nva_class = "nva_class"
    prediction = "prediction"
    patch_scores = "patch_scores"
    patch_scores_legacy = "patch"
    patch_predictions = "patch_predictions"
    side_scores = "side_scores"
    side_predicitions = "side_predictions"
    pixel_scores = "pixel_scores"
    active = "active"
    pixel = "pixel"
    mask = "mask"
    scan = "scan"
    contour_type = "contour_type"
    contour_area_th = "contour_area_th"
    original_width = "original_width"
    original_height = "original_height"
    left_diaphragm_topi = "left_diaphragm_topi"
    right_diaphragm_topi = "right_diaphragm_topi"
    diaphragm_topi = "diaphragm_topi"
    normal = "normal"
    toberead = "toberead"
    extras = "extras"
    prediction_models = "prediction_models"

    # covid related
    covid_risk = "covid_risk"
    covid_percentage = "covid_percentage"
    covid_high = "high"
    covid_low = "low"
    covid_medium = "medium"
    covid_none = "none"
    covid_na = "N/A"

    # after pre-processing
    rmblk_borders = "rmblk_borders"
    pixel_spacing = "pixel_spacing"
    fsnparray = "fsnparray"
    lung_masks = "lung_masks"
    lung_areas = "lung_areas"
    side_borders = "side_borders"
    side_borders_bottom = "side_borders_bottom"

    # tag model keys
    tag_models = "tag_models"
    dependent_abnormality_prediction_models = "dependent_abnormality_prediction_models"
    to_run = "to_run"
    to_show = "to_show"
    has_contour = "has_contour"
    contour_shape = "contour_shape"
    show_contour = "show_contour"

    # pre processing prediction models
    chest = "chest"
    chest_zoom = "chest_zoom"
    fliprot = "fliprot"
    inversion = "inversion"
    # TODO change thorax model's name to thorax_seg
    thorax_seg = "thorax_seg"
    # TODO change diaphragm model's name to diaphragm_seg
    diaphragm_seg = "diaphragm_seg"

    # abnormality prediction models
    atelectasis = "atelectasis"
    bluntedcp = "bluntedcp"
    calcification = "calcification"
    cardiomegaly = "cardiomegaly"
    cavity = "cavity"
    consolidation = "consolidation"
    diaphragm = "diaphragm"
    degenspine = "degenspine"
    emphysema = "emphysema"
    fibrosis = "fibrosis"
    hilarlymphadenopathy = "hilarlymphadenopathy"
    lineandtube = "lineandtube"
    nodule = "nodule"
    opacity = "opacity"
    pacemaker = "pacemaker"
    pleuraleffusion = "pleuraleffusion"
    pneumothorax = "pneumothorax"
    pneumoperitoneum = "pneumoperitoneum"
    ribfracture = "ribfracture"
    scoliosis = "scoliosis"
    trachealshift = "trachealshift"
    tuberculosis = "tuberculosis"
    carina = "carina"
    endotracheal_tube_cls = "endotracheal_tube_cls"
    endotracheal_tube_seg = "endotracheal_tube_seg"
    nasogastric_tube_cls = "nasogastric_tube_cls"
    nasogastric_tube_seg = "nasogastric_tube_seg"
    tracheostomy_tube_seg = "tracheostomy_tube_seg"
    tracheostomy_tube_cls = "tracheostomy_tube_cls"
    reticulonodularpattern = "reticulonodularpattern"
    linearopacity = "linearopacity"
    mediastinalwidening = "mediastinalwidening"
    lung_nodule_malignancy = "lung_nodule_malignancy"
    sutures = "sutures"

    # tag model names
    abnormal = "abnormal"
    covid = "covid"
    breathingtube = "breathingtube"
    gastrictube = "gastrictube"

    # tag types
    side_based = "side_based"
    patch_based = "patch_based"
    scan_224 = "scan_224"
    scan_320 = "scan_320"
    segmentation_224 = "segmentation_224"
    longer_side = "longer_side"
    segmentation_512 = "segmentation_512"
    classification_512 = "classification_512"
    scan_960 = "scan_960"

    # contour types
    convex = "convex"  # for convex hull
    concave = "concave"  # for concave hull
    line = "line"  # for tubes contours
    no_contour = "no_contour"  # tags without contours

    # use cases names
    nva_with_tbr_ce = "nva_with_tbr_ce"
    nva_without_tbr_ce = "nva_without_tbr_ce"
    nva_with_tbr_dev = "nva_with_tbr_dev"
    nva_without_tbr_dev = "nva_without_tbr_dev"
    tb_screening = "tb_screening"
    onlylinesandtubes = "onlylinesandtubes"
    scan_portal = "scan_portal"
    preread_ce = "preread_ce"
    preread_dev = "preread_dev"

    # contour names
    lung_contour = "lung_contour"
    covid_contour = "covid_contour"
    critical_contour = "critical_contour"
    bones_contour = "bones_contour"
    tubes_contour = "tubes_contour"

    # tubes related
    allowed_distance_in_cm = "allowed_distance_in_cm"
    allowed_distance_in_pixels = "allowed_distance_in_pixels"
    min_pix = "min_pix"
    thickness = "thickness"
    offset = "offset"
    tube_presence = "tube_presence"
    tube_position = "tube_position"
    tube_tip = "tube_tip"
    vertical_distance_in_cm = "vertical_distance_in_cm"
    points = "points"


@unique
class PredictionTypes(ConstEnum):
    side_based = [
        i.value
        for i in [
            GeneralKeys.opacity,
            GeneralKeys.nodule,
            GeneralKeys.consolidation,
            GeneralKeys.cavity,
            GeneralKeys.pneumothorax,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.ribfracture,
            GeneralKeys.bluntedcp,
            GeneralKeys.tuberculosis,
            # GeneralKeys.linearopacity,
        ]
    ]
    patch_based = [
        i.value
        for i in [
            GeneralKeys.atelectasis,
            GeneralKeys.calcification,
        ]
    ]
    scan_224 = [
        i.value
        for i in [
            GeneralKeys.trachealshift,
            GeneralKeys.degenspine,
            GeneralKeys.emphysema,
            GeneralKeys.chest,
            GeneralKeys.fliprot,
            GeneralKeys.inversion,
        ]
    ]
    scan_320 = [
        i.value
        for i in [
            GeneralKeys.cardiomegaly,
            GeneralKeys.scoliosis,
            GeneralKeys.pacemaker,
            GeneralKeys.lineandtube,
            GeneralKeys.diaphragm,
            GeneralKeys.sutures,
        ]
    ]
    segmentation_224 = [i.value for i in [GeneralKeys.thorax_seg, GeneralKeys.diaphragm_seg]]
    segmentation_512 = [
        i.value
        for i in [
            GeneralKeys.carina,
            GeneralKeys.endotracheal_tube_seg,
            GeneralKeys.nasogastric_tube_seg,
            GeneralKeys.tracheostomy_tube_seg,
        ]
    ]
    classification_512 = [
        i.value
        for i in [
            GeneralKeys.endotracheal_tube_cls,
            GeneralKeys.nasogastric_tube_cls,
            GeneralKeys.tracheostomy_tube_cls,
            GeneralKeys.chest_zoom,
        ]
    ]
    longer_side = [
        i.value
        for i in [
            GeneralKeys.pleuraleffusion,
            GeneralKeys.bluntedcp,
            GeneralKeys.pneumoperitoneum,
        ]
    ]
    scan_960 = [
        i.value
        for i in [
            GeneralKeys.lung_nodule_malignancy,
            GeneralKeys.mediastinalwidening,
        ]
    ]


class InputSize(ConstEnum):
    side_based = (2, 960, 320)
    patch_based = (6, 320, 320)
    scan_224 = (1, 224, 224)
    scan_320 = (1, 320, 320)
    segmentation_224 = (1, 224, 224)
    longer_side = (2, 960, 320)
    segmentation_512 = (1, 512, 512)
    classification_512 = (1, 512, 512)
    scan_960 = (1, 960, 960)


@unique
class TagKeysToNames(ConstEnum):
    abnormal = "Abnormal"
    atelectasis = "Atelectasis"
    bluntedcp = "Blunted CP"
    calcification = "Calcification"
    cardiomegaly = "Cardiomegaly"
    cavity = "Cavity"
    chest = "Chest"
    consolidation = "Consolidation"
    diaphragm = "Raised/Tented Diaphragm"
    degenspine = "Degenerative Spine condition"
    emphysema = "Emphysema"
    fibrosis = "Fibrosis"
    fliprot = "Fliprot"
    hilarlymphadenopathy = "Hilar Prominence"
    nodule = "Nodule"
    pneumoperitoneum = "Pneumoperitoneum"
    ribfracture = "Rib Fracture"
    opacity = "Opacity"
    peffusion = "Pleural Effusion"
    pneumothorax = "Pneumothorax"
    scoliosis = "Scoliosis"
    trachealshift = "Tracheal Shift"
    tuberculosis = "Tuberculosis"
    covid = "Covid-19"
    breathingtube = "Breathing Tube Placement"
    gastrictube = "Gastric Tube Placement"
    mediastinalwidening = "Mediastinal Widening"
    lung_nodule_malignancy = "Lung Nodule Malignancy"
    sutures = "sutures"


@unique
class ContourShapes(ConstEnum):
    convex = [
        i.value
        for i in [
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.emphysema,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.nodule,
            GeneralKeys.opacity,
            GeneralKeys.ribfracture,
            GeneralKeys.tuberculosis,
            GeneralKeys.reticulonodularpattern,
        ]
    ]
    concave = [
        i.value
        for i in [
            GeneralKeys.pneumothorax,
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.bluntedcp,
        ]
    ]
    line = [
        i.value
        for i in [
            GeneralKeys.breathingtube,
            GeneralKeys.gastrictube,
        ]
    ]
    no_contour = [
        i.value
        for i in [
            GeneralKeys.abnormal,
            GeneralKeys.covid,
            GeneralKeys.cardiomegaly,
            GeneralKeys.calcification,
            GeneralKeys.degenspine,
            GeneralKeys.lineandtube,
            GeneralKeys.pacemaker,
            GeneralKeys.scoliosis,
            GeneralKeys.diaphragm,
            GeneralKeys.atelectasis,
            GeneralKeys.trachealshift,
            GeneralKeys.mediastinalwidening,
            GeneralKeys.lung_nodule_malignancy,
        ]
    ]


class NormalThresholds(ConstEnum):
    abnormal = 0.5
    covid = 0.5
    atelectasis = 1
    bluntedcp = 0.7
    calcification = 1.0
    cardiomegaly = 0.7
    cavity = 1.0
    consolidation = 0.3
    diaphragm = 0.7
    degenspine = 1.0
    emphysema = 0.75
    fibrosis = 0.7
    hilarlymphadenopathy = 0.7
    lineandtube = 0.6
    nodule = 0.5
    opacity = 0.3
    pacemaker = 0.6
    pleuraleffusion = 0.55
    pneumothorax = 0.85
    pneumoperitoneum = 0.75
    ribfracture = 0.75
    scoliosis = 0.9
    trachealshift = 1.0
    tuberculosis = 0.5
    breathingtube = 0.5
    gastrictube = 0.5
    reticulonodularpattern = 1.0
    lung_nodule_malignancy = 0.5
    mediastinalwidening = 0.7
    sutures = 0.6


class AbnormalThresholds(ConstEnum):
    abnormal = 0.5
    covid = 0.6
    atelectasis = 1.0
    bluntedcp = 0.9
    calcification = 1.0
    cardiomegaly = 0.85
    cavity = 1.0
    consolidation = 0.75
    diaphragm = 0.8
    degenspine = 1.0
    emphysema = 0.9
    fibrosis = 0.9
    hilarlymphadenopathy = 0.9
    lineandtube = 0.6
    nodule = 0.7
    opacity = 0.78
    pacemaker = 0.6
    pleuraleffusion = 0.85
    pneumothorax = 0.93
    pneumoperitoneum = 0.75
    ribfracture = 0.9
    scoliosis = 0.98
    trachealshift = 1.0
    tuberculosis = 0.6
    breathingtube = 0.6
    gastrictube = 0.6
    reticulonodularpattern = 1.0
    lung_nodule_malignancy = 0.5
    mediastinalwidening = 0.9
    sutures = 0.6


class ScanThresholds(ConstEnum):
    atelectasis = 0.85
    bluntedcp = 0.9
    calcification = 0.85
    cardiomegaly = 0.8
    cavity = 0.8
    consolidation = 0.5
    diaphragm = 0.75
    degenspine = 0.99
    emphysema = 0.9
    fibrosis = 0.85
    hilarlymphadenopathy = 0.85
    lineandtube = 0.6
    nodule = 0.65
    opacity = 0.5
    pacemaker = 0.6
    pleuraleffusion = 0.8
    pneumothorax = 0.9
    pneumoperitoneum = 0.75
    ribfracture = 0.9
    scoliosis = 0.95
    trachealshift = 0.9
    tuberculosis = 0.5
    endotracheal_tube_cls = 0.6
    endotracheal_tube_seg = 0.5
    nasogastric_tube_cls = 0.8
    nasogastric_tube_seg = 0.5
    tracheostomy_tube_seg = 0.5
    tracheostomy_tube_cls = 0.6
    reticulonodularpattern = 0.9
    carina = 0.5
    lung_nodule_malignancy = 0.5
    mediastinalwidening = 0.9
    sutures = 0.6
    chest = 0.5
    chest_zoom = 0.5
    diaphragm_seg = 0.5
    thorax_seg = 0.5
    fliprot = 0.5
    inversion = 0.5


class PatchThresholds(ConstEnum):
    atelectasis = [0.85, 0.85, 0.85, 0.85, 0.85, 0.85]
    bluntedcp = [1, 1, 0.9, 1, 1, 0.9]
    calcification = [0.75, 0.75, 0.95, 0.75, 0.75, 0.95]
    cavity = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    consolidation = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    fibrosis = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    hilarlymphadenopathy = [0.85, 0.85, 0.85, 0.85, 0.85, 0.85]
    nodule = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
    pneumoperitoneum = [1, 1, 0.75, 1, 1, 0.75]
    reticulonodularpattern = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    ribfracture = [0.65, 0.65, 0.65, 0.65, 0.65, 0.65]
    opacity = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    pleuraleffusion = [1, 1, 0.8, 1, 1, 0.8]
    pneumothorax = [0.85, 0.85, 0.85, 0.85, 0.85, 0.85]
    tuberculosis = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]


class PixelThresholds(ConstEnum):
    carina = 0.4
    endotracheal_tube_seg = 0.5
    nasogastric_tube_seg = 0.5
    bluntedcp = 0.4
    cavity = 0.4
    consolidation = 0.35
    fibrosis = 0.4
    hilarlymphadenopathy = 0.5
    pneumoperitoneum = 0.6
    reticulonodularpattern = 0.5
    ribfracture = 0.6
    opacity = 0.35
    pleuraleffusion = 0.4
    pneumothorax = 0.3
    tuberculosis = 0.4
    nodule = 0.5
    # TODO does tracheostomy_tube_seg not require a pixel threshold? :/


PREPROCESSING_PREDICTION_MODELS = [
    i.value
    for i in [
        GeneralKeys.chest,
        GeneralKeys.chest_zoom,
        GeneralKeys.diaphragm_seg,
        GeneralKeys.thorax_seg,
        GeneralKeys.fliprot,
        GeneralKeys.inversion,
    ]
]
ABNORMALITY_PREDICTION_MODELS = [
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
        GeneralKeys.lineandtube,
        GeneralKeys.nodule,
        GeneralKeys.opacity,
        # GeneralKeys.linearopacity,
        GeneralKeys.pacemaker,
        GeneralKeys.pleuraleffusion,
        GeneralKeys.pneumothorax,
        GeneralKeys.pneumoperitoneum,
        GeneralKeys.ribfracture,
        GeneralKeys.scoliosis,
        GeneralKeys.trachealshift,
        GeneralKeys.tuberculosis,
        GeneralKeys.carina,
        GeneralKeys.endotracheal_tube_cls,
        GeneralKeys.endotracheal_tube_seg,
        GeneralKeys.nasogastric_tube_cls,
        GeneralKeys.nasogastric_tube_seg,
        GeneralKeys.tracheostomy_tube_seg,
        GeneralKeys.tracheostomy_tube_cls,
        GeneralKeys.reticulonodularpattern,
        GeneralKeys.lung_nodule_malignancy,
        GeneralKeys.mediastinalwidening,
        GeneralKeys.sutures,
    ]
]
NEUTRAL_PREDICTION_MODELS = [
    i.value
    for i in [
        GeneralKeys.sutures,
    ]
]
PREDICTION_MODEL_NAMES = PREPROCESSING_PREDICTION_MODELS + ABNORMALITY_PREDICTION_MODELS + NEUTRAL_PREDICTION_MODELS
TAG_MODEL_NAMES = [
    i.value
    for i in [
        GeneralKeys.abnormal,
        GeneralKeys.covid,
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
        GeneralKeys.lineandtube,
        GeneralKeys.nodule,
        GeneralKeys.opacity,
        GeneralKeys.pacemaker,
        GeneralKeys.pleuraleffusion,
        GeneralKeys.pneumothorax,
        GeneralKeys.pneumoperitoneum,
        GeneralKeys.ribfracture,
        GeneralKeys.scoliosis,
        GeneralKeys.trachealshift,
        GeneralKeys.tuberculosis,
        GeneralKeys.breathingtube,
        GeneralKeys.gastrictube,
        GeneralKeys.reticulonodularpattern,
        GeneralKeys.lung_nodule_malignancy,
        GeneralKeys.mediastinalwidening,
        GeneralKeys.sutures,
    ]
]

ALLOWED_PREDICTION_TYPES = [
    i.value
    for i in [
        GeneralKeys.side_based,
        GeneralKeys.patch_based,
        GeneralKeys.scan_224,
        GeneralKeys.scan_320,
        GeneralKeys.segmentation_224,
        GeneralKeys.longer_side,
        GeneralKeys.segmentation_512,
        GeneralKeys.classification_512,
        GeneralKeys.scan_960,
    ]
]
CONTOUR_SHAPES = [i.value for i in [GeneralKeys.convex, GeneralKeys.concave, GeneralKeys.line, GeneralKeys.no_contour]]
USE_CASES = [
    i.value
    for i in [
        GeneralKeys.nva_with_tbr_ce,
        GeneralKeys.nva_without_tbr_ce,
        GeneralKeys.nva_with_tbr_dev,
        GeneralKeys.nva_without_tbr_dev,
        GeneralKeys.tb_screening,
        GeneralKeys.onlylinesandtubes,
        GeneralKeys.scan_portal,
        GeneralKeys.preread_ce,
        GeneralKeys.preread_dev,
    ]
]
CONTOUR_NAMES_COMBINED = [
    i.value
    for i in [
        GeneralKeys.lung_contour,
        GeneralKeys.covid_contour,
        GeneralKeys.critical_contour,
        GeneralKeys.bones_contour,
        GeneralKeys.tubes_contour,
    ]
]
TAGS_TO_COMBINE = [
    i.value
    for i in [
        GeneralKeys.lung_contour,
        GeneralKeys.covid_contour,
        GeneralKeys.critical_contour,
        GeneralKeys.bones_contour,
    ]
]


@unique
class DependentPredictionModels(ConstEnum):
    covid = [
        i.value
        for i in [GeneralKeys.opacity, GeneralKeys.consolidation, GeneralKeys.cavity, GeneralKeys.pleuraleffusion]
    ]
    breathingtube = [
        i.value
        for i in [
            GeneralKeys.carina,
            GeneralKeys.endotracheal_tube_seg,
            GeneralKeys.endotracheal_tube_cls,
            GeneralKeys.tracheostomy_tube_seg,
            GeneralKeys.tracheostomy_tube_cls,
        ]
    ]
    gastrictube = [i.value for i in [GeneralKeys.nasogastric_tube_cls, GeneralKeys.nasogastric_tube_seg]]
    opacity = [
        i.value
        for i in [
            GeneralKeys.opacity,
            GeneralKeys.consolidation,
            GeneralKeys.cavity,
            GeneralKeys.fibrosis,
            GeneralKeys.nodule,
            GeneralKeys.reticulonodularpattern,
        ]
    ]
    abnormal = [
        i
        for i in TAG_MODEL_NAMES
        if i
        not in [
            GeneralKeys.abnormal.value,
            GeneralKeys.covid.value,
            GeneralKeys.breathingtube.value,
            GeneralKeys.gastrictube.value,
            GeneralKeys.lineandtube.value,
            GeneralKeys.pacemaker.value,
            GeneralKeys.atelectasis.value,
            GeneralKeys.calcification.value,
            GeneralKeys.diaphragm.value,
            GeneralKeys.degenspine.value,
            GeneralKeys.emphysema.value,
            GeneralKeys.pneumothorax.value,
            GeneralKeys.pneumoperitoneum.value,
            GeneralKeys.ribfracture.value,
            GeneralKeys.scoliosis.value,
            GeneralKeys.trachealshift.value,
            GeneralKeys.tuberculosis.value,
            GeneralKeys.reticulonodularpattern.value,
            GeneralKeys.sutures.value,
        ]
    ]


@unique
class ContoursToCombine(ConstEnum):
    lung_contour = [
        i.value
        for i in [
            GeneralKeys.opacity,
            GeneralKeys.consolidation,
            GeneralKeys.nodule,
            GeneralKeys.cavity,
            GeneralKeys.fibrosis,
            GeneralKeys.reticulonodularpattern,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.pneumothorax,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.bluntedcp,
        ]
    ]
    covid_contour = [i.value for i in [GeneralKeys.opacity, GeneralKeys.consolidation]]
    bones_contour = [GeneralKeys.ribfracture.value]
    critical_contour = [GeneralKeys.pneumoperitoneum.value]
    tubes_contour = [i.value for i in [GeneralKeys.breathingtube, GeneralKeys.gastrictube]]


# TODO add lung_nodule_malignancy, sutures, mediastinalwidening to reports
@unique
class ReportCombinations(ConstEnum):
    heart = [i.value for i in [GeneralKeys.cardiomegaly]]

    lungs = [
        i.value
        for i in [
            GeneralKeys.opacity,
            GeneralKeys.consolidation,
            GeneralKeys.atelectasis,
            GeneralKeys.cavity,
            GeneralKeys.nodule,
            GeneralKeys.fibrosis,
            GeneralKeys.hilarlymphadenopathy,
            GeneralKeys.calcification,
            GeneralKeys.emphysema,
        ]
    ]
    pleura = [
        i.value
        for i in [
            GeneralKeys.pneumothorax,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.bluntedcp,
            GeneralKeys.pneumoperitoneum,
        ]
    ]
    others = [
        i.value
        for i in [
            GeneralKeys.scoliosis,
            GeneralKeys.degenspine,
            GeneralKeys.trachealshift,
            GeneralKeys.ribfracture,
        ]
    ]
    tubes = [i.value for i in [GeneralKeys.breathingtube, GeneralKeys.gastrictube]]


# TODO add lung_nodule_malignancy, sutures, mediastinalwidening to reports
@unique
class ReportRegions(ConstEnum):
    side = [
        i.value
        for i in [
            GeneralKeys.pneumoperitoneum,
            GeneralKeys.pneumothorax,
            GeneralKeys.pleuraleffusion,
            GeneralKeys.bluntedcp,
            GeneralKeys.ribfracture,
            GeneralKeys.hilarlymphadenopathy,
        ]
    ]
    zone = [
        i.value
        for i in [
            GeneralKeys.opacity,
            GeneralKeys.atelectasis,
            GeneralKeys.cavity,
            GeneralKeys.consolidation,
            GeneralKeys.nodule,
            GeneralKeys.fibrosis,
        ]
    ]
    nolocation = [
        i.value
        for i in [
            GeneralKeys.cardiomegaly,
            GeneralKeys.scoliosis,
            GeneralKeys.calcification,
            GeneralKeys.degenspine,
            GeneralKeys.emphysema,
            GeneralKeys.trachealshift,
        ]
    ]


COMPULSORY_TAGS_ABNORMAL = [
    i.value
    for i in [
        GeneralKeys.cardiomegaly,
        GeneralKeys.consolidation,
        GeneralKeys.nodule,
        GeneralKeys.opacity,
        GeneralKeys.pleuraleffusion,
        GeneralKeys.mediastinalwidening,
    ]
]


# TODO remove this in favor of ContoursToCombine
@unique
class ContourMaskTags(ConstEnum):
    lung = [
        GeneralKeys.opacity,
        GeneralKeys.consolidation,
        GeneralKeys.nodule,
        GeneralKeys.cavity,
        GeneralKeys.fibrosis,
        GeneralKeys.reticulonodularpattern,
        GeneralKeys.hilarlymphadenopathy,
        GeneralKeys.pleuraleffusion,
        GeneralKeys.bluntedcp,
        GeneralKeys.pneumothorax,
    ]
    covid = [GeneralKeys.opacity, GeneralKeys.consolidation]
    critical = [GeneralKeys.pneumoperitoneum]
    bones = [GeneralKeys.ribfracture]


contour_tags_ids = {
    GeneralKeys.opacity.value: 0,
    GeneralKeys.consolidation.value: 1,
    GeneralKeys.nodule.value: 2,
    GeneralKeys.cavity.value: 3,
    GeneralKeys.fibrosis.value: 4,
    GeneralKeys.reticulonodularpattern.value: 5,
    GeneralKeys.hilarlymphadenopathy.value: 6,
    GeneralKeys.linearopacity.value: 7,
    GeneralKeys.pleuraleffusion.value: 8,
    GeneralKeys.bluntedcp.value: 9,
    GeneralKeys.pneumothorax.value: 10,
    GeneralKeys.pneumoperitoneum.value: 11,
    GeneralKeys.ribfracture.value: 12,
}

contour_reverse_tags_ids = dict((contour_tags_ids[x], x) for x in contour_tags_ids)


class DownSampleSizes:
    downsample_sizes = [224, 320, 512, 960]
