from collections import OrderedDict
from typing import Dict, List, Union

import numpy as np

from qxr_utils.postprocess.text_report.report_constants import default_report_str, template_normals, zone_dict
from qxr_utils.tag_config.constants import (
    COMPULSORY_TAGS_ABNORMAL,
    TAG_MODEL_NAMES,
    GeneralKeys,
    PredictionTypes,
    ReportCombinations,
    ReportRegions,
)

# predsdict format -> {abn: {prediction, patch_prediction}}


def return_zone_text_from_patch(patch_predictions: Union[List[int], np.ndarray]) -> str:
    """returns the zone text string for a patch prediction

    Args:
        patch_predictions (Union[List[int], np.ndarray]): list of length 6 integer or numpy array

    Returns:
        str: string output of zone report
    """
    out = ""
    for pred in patch_predictions:
        out += str(pred * 1)
    assert len(out) == 6
    return zone_dict[out]


def return_side_text_from_patch(patch_predictions: Union[List[int], np.ndarray]) -> str:
    """Returns the side text for a prediction set

    Args:
        patch_predictions (Union[List[int], np.ndarray]): list of length 6 integer or numpy array

    Returns:
        str: string output of side on the report
    """
    if not max(patch_predictions):
        return None
    elif max(patch_predictions[:3]) and not max(patch_predictions[3:]):
        return "on the left"
    elif not max(patch_predictions[:3]) and max(patch_predictions[3:]):
        return "on the right"
    else:
        return "on both sides"


def get_tube_text(preds_dict: dict) -> dict:
    """Function generates the lines which go into tubes reporting

    Args:
        preds_dict (dict): Standard Predictions dictionary format which has a tag and its predictions

    Returns:
        dict: dictionary with tag name and its corresponding report line
    """
    tube_tags = [GeneralKeys.breathingtube.value, GeneralKeys.gastrictube.value]
    tag_lines = {}
    for tag in tube_tags:
        if tag in preds_dict and preds_dict[tag][GeneralKeys.extras.value][GeneralKeys.tube_presence.value]:
            tag_lines[tag] = default_report_str[tag]
            if preds_dict[tag][GeneralKeys.extras.value][GeneralKeys.tube_position.value]:
                tag_lines[tag] += " out of position\n"
            else:
                tag_lines[tag] += " in position\n"
    return tag_lines


def get_side_text_all_tags(preds_dict: dict) -> dict:
    """Function generates the lines which go into side level tag reporting.
    example: pneumothorax, pleural effusion

    Args:
        preds_dict (dict): Standard Predictions dictionary format which has a tag and its predictions

    Returns:
        dict: dictionary with tag name and its corresponding report line
    """
    tag_lines = {}
    for tag in ReportRegions.side.value:
        if tag in preds_dict and preds_dict[tag][GeneralKeys.prediction.value]:
            tag_lines[tag] = default_report_str[tag]
            side_info = return_side_text_from_patch(preds_dict[tag][GeneralKeys.patch_predictions.value])
            if side_info is not None:
                tag_lines[tag] = tag_lines[tag] + " " + side_info
    return tag_lines


def get_zone_text_all_tags(preds_dict: dict) -> dict:
    """Function generates the lines which go into zone level tag reporting.
    example: opacity, consolidation, nodule

    Args:
        preds_dict (dict): Standard Predictions dictionary format which has a tag and its predictions

    Returns:
        dict: dictionary with tag name and its corresponding report line
    """
    tag_lines = {}
    for tag in ReportRegions.zone.value:
        if tag in preds_dict and preds_dict[tag][GeneralKeys.prediction.value]:
            tag_lines[tag] = default_report_str[tag]
            zone_info = return_zone_text_from_patch(preds_dict[tag][GeneralKeys.patch_predictions.value])
            if zone_info is not None:
                tag_lines[tag] += " in " + zone_info
    return tag_lines


def get_no_location_text_all_tags(preds_dict: dict) -> dict:
    """Function generates the lines which go into no location tag reporting.
    example: cardiomegaly, scoliosis

    Args:
        preds_dict (dict): Standard Predictions dictionary format which has a tag and its predictions

    Returns:
        dict: dictionary with tag name and its corresponding report line
    """
    tag_lines = {}
    for tag in ReportRegions.nolocation.value:
        if tag in preds_dict and preds_dict[tag][GeneralKeys.prediction.value]:
            tag_lines[tag] = default_report_str[tag]
    return tag_lines


def consolidate_all_findings(preds_dict: dict) -> dict:
    """Consolidates all the findings from various ways in which they are obtained such as:
    zone level, side level, no-location type, tubes

    Args:
        preds_dict (dict): Standard Predictions dictionary format which has a tag and its predictions

    Returns:
        dict: findings dictionary which has every tag as a key and its report as its value
    """
    all_findings: Dict[str, str] = {}
    all_findings = {**all_findings, **get_side_text_all_tags(preds_dict)}
    all_findings = {**all_findings, **get_zone_text_all_tags(preds_dict)}
    all_findings = {**all_findings, **get_tube_text(preds_dict)}
    all_findings = {**all_findings, **get_no_location_text_all_tags(preds_dict)}
    return all_findings


def give_heart_region_report(all_findings: dict, preds_dict: dict) -> str:
    """Returns the heart region report as string

    Args:
        all_findings (dict): All tag findings, each key is a tag, each value is a string
        preds_dict (dict): Standard Predictions dictionary format which has a tag and its predictions

    Returns:
        str: output report in string format for heart abnormalities
    """
    findings = ""
    for abn in ReportCombinations.heart.value:
        if abn in all_findings:
            findings = all_findings[abn] + "\n"
    if len(findings) == 0:
        if ReportCombinations.heart.value[0] in preds_dict:
            findings = "Heart appears normal\n"
        else:
            findings = "This configuration of qXR cannot comment on heart size"
    return findings


def give_lung_region_report(all_findings: dict, preds_dict: dict) -> str:
    """Returns the lung region report as string

    Args:
        all_findings (dict): All tag findings, each key is a tag, each value is a string
        preds_dict (dict): Standard Predictions dictionary format which has a tag and its predictions

    Returns:
        str: output report in string format for lung region tags
    """
    findings = ""
    for abn in ReportCombinations.lungs.value:
        if abn in all_findings:
            findings += all_findings[abn] + "\n"
    return findings


def give_pleura_region_report(all_findings: dict, preds_dict: dict) -> str:
    """Returns the pleura region report as string

    Args:
        all_findings (dict): All tag findings, each key is a tag, each value is a string
        preds_dict (dict): Standard Predictions dictionary format which has a tag and its predictions

    Returns:
        str: output report in string format for pleural tags
    """
    findings = ""
    for abn in ReportCombinations.pleura.value:
        if abn in all_findings:
            findings += all_findings[abn] + "\n"
    return findings


def give_other_region_report(all_findings: dict, preds_dict: dict) -> str:
    """Returns the other region's report as string

    Args:
        all_findings (dict): All tag findings, each key is a tag, each value is a string
        preds_dict (dict): Standard Predictions dictionary format which has a tag and its predictions

    Returns:
        str: output report in string format for other tags
    """
    findings = ""
    for abn in ReportCombinations.others.value:
        if abn in all_findings:
            findings += all_findings[abn] + "\n"
    return findings


def give_tubes_tag_report(all_findings: dict, preds_dict: dict) -> str:
    """Returns the tubes report as string

    Args:
        all_findings (dict): All tag findings, each key is a tag, each value is a string
        preds_dict (dict): Standard Predictions dictionary format which has a tag and its predictions

    Returns:
        str: output report in string format for tubes tags
    """
    findings = ""
    count = 0
    for abn in ReportCombinations.tubes.value:
        if abn in all_findings:
            findings += all_findings[abn]
        if abn in preds_dict:
            count += 1
    if len(findings) == 0 and count == 0:
        findings = "This configuration of qXR cannot comment on the presence/position of tubes"
    return findings


def get_study_impression(preds_dict: dict) -> str:
    """Function outputs the study impression based on the prediction of abnormal tag

    Args:
        preds_dict (dict): Standard Predictions dictionary format which has a tag and its predictions

    Returns:
        str: impression in string format
    """
    impression = ""
    if GeneralKeys.abnormal.value in preds_dict:
        if preds_dict[GeneralKeys.abnormal.value][GeneralKeys.prediction.value]:
            impression = "Abnormal Study"
        else:
            impression = "No significant abnormality detected"
    compulsary_tags_count = 0
    for tag in COMPULSORY_TAGS_ABNORMAL:
        if tag in preds_dict:
            compulsary_tags_count += 1
    if compulsary_tags_count != len(COMPULSORY_TAGS_ABNORMAL):
        impression = "This configuration of qXR cannot determine if the overall study is normal"
    return impression


def check_essential_lung_tags_execution(preds_dict: dict) -> bool:
    """function checks if all the essential tags are present during the execution

    Args:
        preds_dict (dict): Standard Predictions dictionary format which has a tag and its predictions

    Returns:
        bool: True/False based on if the essential tags are all present in the preds dict
    """
    essential_tags = [GeneralKeys.opacity.value, GeneralKeys.consolidation.value, GeneralKeys.nodule.value]
    tag_count = 0
    for tag in essential_tags:
        if tag in preds_dict:
            tag_count += 1
    return tag_count == len(essential_tags)


def get_study_findings(preds_dict: dict) -> dict:
    """Function outputs all the findings of various fields in the lung,
    also certain critical abnormalities in its raw format.

    Args:
        preds_dict (dict): Standard Predictions dictionary format which has a tag and its predictions

    Returns:
        dict: dictionary of various fields in the lung, each having a string representing its report
    """
    findings = OrderedDict()
    preds_dict = change_patch_predictions(preds_dict)
    tag_findings = consolidate_all_findings(preds_dict)
    lung_findings = give_lung_region_report(tag_findings, preds_dict)
    pleura_findings = give_pleura_region_report(tag_findings, preds_dict)
    if len(lung_findings) == 0:
        if check_essential_lung_tags_execution:
            findings["lung"] = pleura_findings + "\n"
        else:
            findings["lung"] = "This configuration of qXR can't determine the presence of lung infilrates"
            findings["pleura"] = pleura_findings + "\n"
    else:
        findings["lung"] = lung_findings + "\n"
        findings["pleura"] = pleura_findings + "\n"

    findings["heart"] = give_heart_region_report(tag_findings, preds_dict)
    other_findings = give_other_region_report(tag_findings, preds_dict)
    if len(other_findings) > 0:
        findings["others"] = "Other Findings:\n" + other_findings + "\n"

    if GeneralKeys.covid_risk.value in preds_dict:
        risk = preds_dict[GeneralKeys.covid_risk.value][GeneralKeys.thresholded_score.value]
        findings["covid"] = "Covid-19 Risk: {}\n".format(risk)

    if (
        GeneralKeys.tuberculosis.value in preds_dict
        and preds_dict[GeneralKeys.tuberculosis.value][GeneralKeys.prediction.value]
    ):
        findings["tb"] = "\nTuberculosis screen advised\n"

    tube_findings = give_tubes_tag_report(tag_findings, preds_dict)
    if len(tube_findings) > 0:
        findings["tubes"] = "\nPlacement of Tubes:\n" + tube_findings + "\n"

    return findings


def get_study_report(preds_dict: dict) -> dict:
    """Function outputs the final report as findings and impression

    Args:
        preds_dict (dict): Standard Predictions dictionary format which has a tag and its predictions

    Returns:
        dict: dictionary with keys labelled as findings and impression
    """
    template = "template1"
    findings = get_study_findings(preds_dict)
    impression = get_study_impression(preds_dict)
    if preds_dict[GeneralKeys.nva_class.value] == GeneralKeys.normal.value:
        findings = template_normals[template]
        impression = "No significant abnormality detected"
    elif preds_dict[GeneralKeys.nva_class.value] == GeneralKeys.toberead.value:
        findings = {"result": "X-Ray to be read"}
        impression = "To be read by a radiologist"
    return {"findings": findings, "impression": impression}


def change_patch_predictions(preds_dict: dict) -> dict:
    """Function to change patch predictions when multiple overlapping abnormalities are present
    in the same predictions dict

    Args:
        preds_dict (dict): Standard Predictions dictionary format which has a tag and its predictions

    Returns:
        dict: Modified predictions dictionary
    """

    def change_wrt_tags(preds_dict, super_tag, sub_tags):
        if super_tag in preds_dict:
            for sub_tag in sub_tags:
                if sub_tag in preds_dict and preds_dict[sub_tag][GeneralKeys.prediction.value]:
                    preds_dict[super_tag][GeneralKeys.prediction.value] = False
                    for i in range(len(preds_dict[sub_tag][GeneralKeys.patch_predictions.value])):
                        preds_dict[super_tag][GeneralKeys.patch_predictions.value][i] = False
        return preds_dict

    super_tag = GeneralKeys.atelectasis.value
    sub_tags = [GeneralKeys.pneumothorax.value, GeneralKeys.fibrosis.value]
    preds_dict = change_wrt_tags(preds_dict, super_tag, sub_tags)

    super_tag = GeneralKeys.bluntedcp.value
    sub_tags = [GeneralKeys.pleuraleffusion.value]
    preds_dict = change_wrt_tags(preds_dict, super_tag, sub_tags)

    super_tag = GeneralKeys.calcification.value
    sub_tags = [GeneralKeys.nodule.value]
    preds_dict = change_wrt_tags(preds_dict, super_tag, sub_tags)

    super_tag = GeneralKeys.hilarlymphadenopathy.value
    sub_tags = [
        GeneralKeys.opacity.value,
        GeneralKeys.consolidation.value,
        GeneralKeys.nodule.value,
        GeneralKeys.fibrosis.value,
        GeneralKeys.cavity.value,
    ]
    if super_tag in preds_dict:
        tag_count = 0
        for sub_tag in sub_tags:
            if sub_tag in preds_dict and preds_dict[sub_tag][GeneralKeys.prediction.value]:
                tag_count += 1
        if tag_count > 1:
            preds_dict[super_tag][GeneralKeys.prediction.value] = False

    return preds_dict
