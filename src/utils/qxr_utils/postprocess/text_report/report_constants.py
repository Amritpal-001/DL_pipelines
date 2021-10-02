from qxr_utils.tag_config.constants import GeneralKeys, PredictionTypes

# zone_dict is a dictionary which gives out the location text for a string encoding of patch predictions output
zone_dict = {
    "000000": None,
    "000001": "right lower zone",
    "000010": "right middle zone",
    "000011": "right middle and lower zones",
    "000100": "right upper zone",
    "000101": "right upper and lower zones",
    "000110": "right upper and middle zones",
    "000111": "right lung",
    "001000": "left lower zone",
    "001001": "bilateral lower zones",
    "001010": "left lower and right mid zones",
    "001011": "bilateral lower zones and right mid zone",
    "001100": "left lower and right upper zones",
    "001101": "bilateral lower zones and right upper zone",
    "001110": "left lower, right upper and right mid zones",
    "001111": "right lung and left lower zone",
    "010000": "left middle zone",
    "010001": "left middle and right lower zones",
    "010010": "bilateral middle zones",
    "010011": "bilateral middle zones and right lower zone",
    "010100": "left mid zone and right upper zone",
    "010101": "left mid, right upper and right lower zone",
    "010110": "bilateral middle zones and right upper zone",
    "010111": "right lung and left middle zone",
    "011000": "left mid and lower zones",
    "011001": "bilateral lower zones and left mid zone",
    "011010": "bilateral mid zones and left lower zone",
    "011011": "bilateral mid and lower zones",
    "011100": "left mid, left lower and right upper zones",
    "011101": "bilateral lower zones, left mid and right upper zone",
    "011110": "bilateral mid zones, left lower and right upper zone",
    "011111": "right lung and left mid, lower zones",
    "100000": "left upper zone",
    "100001": "left upper, right lower zone",
    "100010": "left upper, right middle zone",
    "100011": "left upper, right middle and lower zones",
    "100100": "bilateral upper zone",
    "100101": "bilateral upper and right lower zones",
    "100110": "bilateral upper and right middle zones",
    "100111": "right lung and left upper zone",
    "101000": "left upper, lower zones",
    "101001": "left upper zone and bilateral lower zones",
    "101010": "left upper, left lower and right mid zones",
    "101011": "bilateral lower zones, left upper and right mid zone",
    "101100": "left lower and bilateral upper zones",
    "101101": "bilateral upper and lower zones",
    "101110": "bilateral upper zones, left lower and right mid zones",
    "101111": "right lung and left upper, lower zones",
    "110000": "left upper and mid zones",
    "110001": "left upper, left middle and right lower zones",
    "110010": "bilateral middle zones and left upper zone",
    "110011": "bilateral middle zones, right lower and left upper zone",
    "110100": "bilateral upper zones and left mid zone",
    "110101": "bilateral upper zones, left mid and right lower zone",
    "110110": "bilateral upper and mid zones",
    "110111": "bilateral upper and mid zones along with right lower zone",
    "111000": "left lung",
    "111001": "left lung and right lower zone",
    "111010": "left lung and right mid zone",
    "111011": "left lung and right mid, lower zones",
    "111100": "left lung and right upper zone",
    "111101": "left lung, right upper and lower zones",
    "111110": "left lung, right upper and mid zones",
    "111111": "both lung fields",
}

# default report str gives the default report, this is used in combination with zone dictionary to give valid outputs
default_report_str = {
    GeneralKeys.atelectasis.value: "Atelectasis is observed",
    GeneralKeys.opacity.value: "Opacity is observed",
    GeneralKeys.consolidation.value: "Inhomogeous Opacity, probable Consolidation is observed",
    GeneralKeys.cavity.value: "Inhomogeous Opacity, probable Cavitation is observed",
    GeneralKeys.nodule.value: "Nodular Opacity is observed",
    GeneralKeys.fibrosis.value: "Fibrotic changes are observed",
    GeneralKeys.hilarlymphadenopathy.value: "Hilum appears abnormal",
    GeneralKeys.cardiomegaly.value: "The Heart is enlarged. Cardiomegaly",
    GeneralKeys.diaphragm.value: "Raising/Tenting observed on the Hemi-Diaphragm",
    GeneralKeys.scoliosis.value: "Scoliosis is observed",
    GeneralKeys.degenspine.value: "Degenerative Spine changes are observed",
    GeneralKeys.emphysema.value: "Probable Emphysema is observed",
    GeneralKeys.trachealshift.value: "Tracheal Shift is observed",
    GeneralKeys.pneumothorax.value: "Pneumothorax is observed",
    GeneralKeys.pleuraleffusion.value: "Pleural Effusion is observed",
    GeneralKeys.bluntedcp.value: "Blunting of CP angle is observed",
    GeneralKeys.calcification.value: "Calcification is noted",
    GeneralKeys.ribfracture.value: "Rib Fracture is found",
    GeneralKeys.breathingtube.value: "Breathing tube is seen",
    GeneralKeys.gastrictube.value: "Gastric Tube is seen",
    GeneralKeys.lineandtube.value: "Lines/Tubes present in the CXR",
    GeneralKeys.pacemaker.value: "Pacemaker is seen",
    GeneralKeys.pneumoperitoneum.value: "Pneumoperitoneum is observed below the Diaphragm",
    GeneralKeys.reticulonodularpattern.value: "Reticular Opacities are observed",
}

# various hospitals ask for various normal template settings. These are 2 of them
template_normals = {
    "template1": {
        "lung": "No significant lung field abnormality detected.\n",
        "pleura": "Pleura appears normal.\n",
        "heart": "Heart appears normal.\n",
        "mediastinum": "The mediastinum is within normal limits.\n",
    },
    "template2": {
        "lung": "Lung fields are clear.\n",
        "heart": "Heart is not enlarged.\n",
        "diaphragm": "Diaphragm is intact.\n",
        "verdict": "Impression:\n\nEsentially Normal Chest Findings.\n",
    },
}
