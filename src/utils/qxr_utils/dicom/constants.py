class MetadataConstants:
    SeriesDescription = "SeriesDescription"
    ViewPosition = "ViewPosition"
    AcquisitionDeviceProcessingDescription = "AcquisitionDeviceProcessingDescription"
    ProtocolName = "ProtocolName"
    Modality = "Modality"
    PerformedProcedureStepDescription = "PerformedProcedureStepDescription"
    Manufacturer = "Manufacturer"
    BodyPartExamined = "BodyPartExamined"


class ChestMetadataConstants:
    # holds the relevant metadata tags which are to be checked for chest/ relevant words presence
    manufacturer_metadata_tags = {
        "agfa": [MetadataConstants.SeriesDescription],
        "carestream": [MetadataConstants.ViewPosition, MetadataConstants.SeriesDescription],
        "fuji": [MetadataConstants.AcquisitionDeviceProcessingDescription],
        "rehbein": [MetadataConstants.SeriesDescription],
        "canon": [MetadataConstants.ProtocolName, MetadataConstants.SeriesDescription],
        "allengers": [MetadataConstants.SeriesDescription],
        "kodak": [MetadataConstants.SeriesDescription],
        "e-com": [MetadataConstants.SeriesDescription],
        "philips": [MetadataConstants.ViewPosition],
        "konica": [MetadataConstants.SeriesDescription],
        "varian": [MetadataConstants.ViewPosition],
        "3disc": [MetadataConstants.SeriesDescription],
        "samsung": [MetadataConstants.ProtocolName],
        "gehealth": [MetadataConstants.AcquisitionDeviceProcessingDescription],
        "gemedical": [MetadataConstants.AcquisitionDeviceProcessingDescription],
        "default": [MetadataConstants.ViewPosition, MetadataConstants.SeriesDescription],
    }

    accepted_modalities = ["cr", "dx", "dr"]

    dicom_tags_to_check = [
        MetadataConstants.SeriesDescription,
        MetadataConstants.PerformedProcedureStepDescription,
        MetadataConstants.ViewPosition,
        MetadataConstants.AcquisitionDeviceProcessingDescription,
        MetadataConstants.ProtocolName,
    ]

    chest_key_words = ["chest", "thorax", "cht"]

    # words indicative of non-chest cxrs
    negwords = [
        "obl",
        "lat",
        "decub",
        "naso",
        "lao",
        "lpo",
        "rao",
        "rpo",
        "rld",
        "spine",
        "should",
        "abdo",
        "clav",
        "lodo",
        "scapul",
        "apico",
    ]
