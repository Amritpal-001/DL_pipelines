import logging
import os
from string import punctuation
from typing import Any

import numpy as np
import pydicom as dicom
import SimpleITK as sitk

from qxr_utils.exceptions import InvalidPixelData
from qxr_utils.image import transforms as tf

logger = logging.getLogger("DICOM HANDLER")
sitk_version = 1
if hasattr(sitk, "__version__"):
    sitk_version = int(getattr(sitk, "__version__").split(".")[0])


def is_dicom(dcm_path: str, force: bool = False) -> bool:
    """
    Checks if the dicom is valid by reading 128-byte DICOM preamble
    Args:
        dcm_path: path to dicom
        force: Flag to force reading of a file even if no header is found.

    Returns:
        True if it's a valid dicom, else False

    """
    assert os.path.exists(dcm_path)
    with open(dcm_path, "rb") as dcm:
        try:
            # https://github.com/pydicom/pydicom/blob/056c0f18d3e51e5626c774bb3a5a17d11e562a1f/pydicom/filereader.py
            # force = False since we don't want reading of a file even if no header is found
            preamble = dicom.filereader.read_preamble(dcm, force=force)
            if preamble is not None:
                return True
            else:
                return False
        except Exception as e:
            logger.exception(e)
            return False


def get_metadata_from_dicom(dcm_path: str) -> dicom.dataset.FileDataset:
    """
    extract the metadata from a dicom file
    Args:
        dcm_path: path to dicom

    Returns:
        dicom metadata as dicom dataset
    """
    assert os.path.exists(dcm_path)
    try:
        dcm_meta = dicom.read_file(dcm_path, stop_before_pixels=True, force=True)
        return dcm_meta
    except Exception as e:
        logger.exception("unable to read metadata from dicom")
        raise RuntimeError("unable to read metadata from dicom") from e


def get_array_from_dicom(dcm_path: str) -> np.ndarray:
    """
    Reads the pixel array from a dicom, modifies it appropriately for the deep learning models to consume
    Args:
        dcm_path: path to dicom

    Returns:
        Pixel array from dicom

    """
    try:
        dcm_meta = get_metadata_from_dicom(dcm_path)
        mode = dcm_meta.get("PhotometricInterpretation")
        try:
            dicom_image = sitk.GetArrayFromImage(sitk.ReadImage(dcm_path))
        except Exception as e:
            logger.exception("unable to read pixel data")
            raise InvalidPixelData(e)
        if sitk_version >= 2 and mode == "MONOCHROME1":
            dicom_image = dicom_image.max() - dicom_image
        try:
            if hasattr(dcm_meta, "WindowWidth") and hasattr(dcm_meta, "WindowCenter"):
                if isinstance(getattr(dcm_meta, "WindowWidth"), dicom.multival.MultiValue):
                    ww = int(int(getattr(dcm_meta, "WindowWidth")[0]) / 2)
                else:
                    ww = int(int(getattr(dcm_meta, "WindowWidth")) / 2)
                if isinstance(getattr(dcm_meta, "WindowCenter"), dicom.multival.MultiValue):
                    wc = int(getattr(dcm_meta, "WindowCenter")[0])
                else:
                    wc = int(getattr(dcm_meta, "WindowCenter"))
                low_clip = max(np.min(dicom_image), wc - ww)
                high_clip = min(np.max(dicom_image), wc + ww)
                dicom_image = np.clip(dicom_image, low_clip, high_clip)
        except Exception as e:
            # Not doing windowing because of error in dicom
            logger.exception(e)
            dicom_image = dicom_image

        # mean of array using photometric interpretation
        if mode == "MONOCHROME1":
            dicom_image = dicom_image.max() - dicom_image
        # TODO get dim 0 only if dicom image is of the shape (1,x,y)
        im_array = dicom_image[0]
        if im_array.ndim == 3 and im_array.shape[0] == 1:
            im_array = im_array[0]
        elif im_array.ndim == 3:
            im_array = tf.mean_with_nonzero_std(im_array)
        return im_array
    except InvalidPixelData as e:
        logger.exception("Invalid pixel data")
        raise e
    except Exception as e:
        logger.exception("unable to read pixel data from dicom")
        raise RuntimeError("unable to read pixel data from dicom") from e


def dicom_dataset_to_dict(dicom_header: dicom.dataset.FileDataset) -> dict:
    """
    Converts DicomDataset class to dict of tag, value.
    Args:
        dicom_header: dicom dataset

    Returns: dict of tag and value

    """
    dicom_dict = {}
    try:
        repr(dicom_header)
        for dicom_value in dicom_header.values():
            if dicom_value.tag == (0x7FE0, 0x0010):
                # discard pixel data
                continue
            key = dicom_value.name
            if key == "Private Creator" or len(key) == 0:
                # fixing the fuck up caused by private creator, for eg - dmims dicoms
                key = key + "_" + str(dicom_value.tag).translate({ord(x): "" for x in punctuation})
            if type(dicom_value.value) == dicom.dataset.Dataset:
                dicom_dict[key] = dicom_dataset_to_dict(dicom_value.value)
            else:
                v = _convert_value(dicom_value.value)
                dicom_dict[key] = v
    except Exception as e:
        logging.exception(e)
        pass
    return dicom_dict


def _sanitise_unicode(s: str) -> str:
    """
    removes unicode
    Args:
        s: string

    Returns: string stripped of unicode

    """
    return s.replace(u"\u0000", "").strip()


def _convert_value(v: Any) -> Any:
    """
    converts values in a dicom dataset
    Args:
        v: value

    Returns: value casted as appropriate type

    """
    t = type(v)
    if t in (list, int, float):
        cv = v
    elif t == str:
        cv = _sanitise_unicode(v)
    elif t == bytes:
        s = v.decode("ascii", "replace")
        cv = _sanitise_unicode(s)
    elif t == dicom.valuerep.DSfloat:
        cv = float(v)
    elif t == dicom.valuerep.IS:
        cv = int(v)
    elif t == dicom.valuerep.PersonName:
        cv = str(v)
    else:
        cv = repr(v)
    return cv


def _age(st: str) -> int:
    """
    extracts age from dicom metadata value
    Args:
        st: string from dicom metadata

    Returns: age as int

    """
    s = st.lower()
    try:
        if "y" in s:
            return int(s[:-1])
        elif ("d" in s) or ("m" in s):
            return 0
        else:
            return -1
    except Exception as e:
        logging.exception(e)
        return -1


def get_metadata_as_dict(dcmpath: str) -> dict:
    """
    returns dicom metata as dict
    Args:
        dcmpath: path to dicom file

    Returns: dicom metadata as a dict

    """
    try:
        dcm_val = dicom.dcmread(dcmpath, force=True, stop_before_pixels=True)
        dcm_metadata = dicom_dataset_to_dict(dcm_val)
        return dcm_metadata
    except Exception as e:
        print(dcmpath, e)
        return {}


def get_pixel_spacing(metadata: dict):
    """Generates pixel spacing from metadata dictonary if PixelSpacing/ImagerPixelSpacing are available. If unavailable, returns -1.
    Note: Assumes pixel spacing in x and y dimensions is equal.

    Args:
        metadata ([dicom.dataset.FileDataset]): dicom metadata

    Returns:
        [float]: pixel spacing
    """

    px = float(-1)
    try:
        if hasattr(metadata, "PixelSpacing"):
            PixelSpacing = metadata.get("PixelSpacing")[0]
            if PixelSpacing is not None:
                px = float(PixelSpacing)
        else:
            px = -1
    except Exception as e:
        logging.exception(e)
        px = -1
    return px
