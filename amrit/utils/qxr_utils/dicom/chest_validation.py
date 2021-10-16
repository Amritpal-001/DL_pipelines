from typing import List

import pydicom as dicom
from pydantic import BaseModel

from qxr_utils.dicom.constants import ChestMetadataConstants as CMC
from qxr_utils.dicom.constants import MetadataConstants as MC


class ChestMetadata(BaseModel):
    """
    #TODO TBR to write a detailed desciption of each of the variables
    """

    valid: bool = True
    body: bool = False
    view: bool = False
    nobody: bool = False
    modality: bool = True


class ChestImage(BaseModel):
    """
    #TODO TBR to write detailed description of each of the variables
    """

    valid: bool = False
    result: float = 0
    threshold: float = 0.5
    message: str = "Invalid dicom file"


def get_manufacturer_tags(manufacturer: str) -> List:
    """
    Gets the tags relevant to manufacturer from ChestMetadataConstants
    Args:
        manufacturer: name of the manufacturer

    Returns:
        metadata tag which holds the information relevant to chest

    """
    manufacturer = manufacturer.lower().replace(" ", "")
    for _manufacturer in CMC.manufacturer_metadata_tags:
        if _manufacturer in manufacturer:
            return CMC.manufacturer_metadata_tags[_manufacturer]
    return CMC.manufacturer_metadata_tags["default"]


def is_bodypart_chest(bodypart: str) -> bool:
    """
    check if the body part is chest or not
    Args:
        bodypart: body part examined

    Returns:
        True if the body part is chest, else false

    """
    return bodypart.lower() in CMC.chest_key_words


def validate_chest_metadata(metadata: dicom.dataset.FileDataset) -> ChestMetadata:
    """
    #TODO TBR to write a detailed description
    Args:
        metadata: dicom metadata

    Returns:
        Instance of a ChestMetadata object

    """
    out = ChestMetadata()
    manufacturer = str(getattr(metadata, MC.Manufacturer)) if hasattr(metadata, MC.Manufacturer) else "default"
    manufacturer_tags = get_manufacturer_tags(manufacturer)
    modality = ""
    if hasattr(metadata, MC.Modality):
        modality = str(getattr(metadata, MC.Modality)).lower()
    if modality not in modality:
        out.valid = False
        out.modality = False
        return out
    if hasattr(metadata, MC.BodyPartExamined):
        body_part = str(getattr(metadata, MC.BodyPartExamined)).lower()

        if is_bodypart_chest(body_part):
            check_neg = False
            for manufacturer_tag in manufacturer_tags:
                if hasattr(metadata, manufacturer_tag):
                    seriesdata = str(getattr(metadata, manufacturer_tag)).lower()
                    for negword in CMC.negwords:
                        if negword in seriesdata:
                            check_neg = True
            if check_neg:
                out.valid = False
                out.view = True
        else:
            check_neg = False
            check_chest = False
            for dicom_tag in CMC.dicom_tags_to_check:
                if hasattr(metadata, dicom_tag):
                    seriesdata = str(getattr(metadata, dicom_tag)).lower()
                    if is_bodypart_chest(seriesdata):
                        check_chest = True
                    for negword in CMC.negwords:
                        if negword in seriesdata:
                            check_neg = True
            if not check_neg and check_chest:
                out.valid = True
                out.view = False
            elif len(body_part) > 1:
                out.valid = False
                out.body = True
            else:
                out.valid = False
                out.nobody = True
    else:
        out.valid = False
        out.nobody = True

    return out


def validate_chest_image(
    model_result: float, metadata: ChestMetadata, threshold: float = 0.5, unreadable_dicom: bool = False
) -> ChestImage:
    """
    validates if a dicom is chest image or not
    Args:
        model_result: result of chest vs non-chest model
        metadata: chest relevant metadata extracted from dicom metadata
        threshold: threshold for chest vs non-chest model
        unreadable_dicom: true if the dicom pixel data is readable

    Returns:
        Instance of ChestImage
    """
    output = ChestImage()
    if not unreadable_dicom:
        valid = model_result >= threshold
        msg = "Valid CXR" if valid else "Invalid CXR"

        output.valid = valid
        output.result = model_result
        output.threshold = threshold
        output.message = msg

        if not metadata.modality:
            output.valid = False
            output.message = "Dicom Modality is not CR/DX/DR"
        elif metadata.view:
            output.valid = False
            output.message = "DICOM Metadata Validation Failed"
        elif metadata.body:
            output.valid = False
            output.message = "DICOM Metadata Validation Failed"
        elif not output.valid:
            output.message = "DICOM Metadata Validation Failed"
        return output
    else:
        return output
