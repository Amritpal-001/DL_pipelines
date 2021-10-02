import numpy as np
import SimpleITK as sitk
import cv2
import os
import json

brain_window = (80, 40)
blood_window = (175, 50)
bone_window = (3000, 500)


def ct_window(array, window_param):
    """Return CT window transform for given width and level."""
    window_width, window_level = window_param[0], window_param[1]
    low = window_level - window_width / 2
    high = window_level + window_width / 2
    return np.clip(array, low, high)


def get_sitkimage(sitk_img, new_array):
    return_image = sitk.GetImageFromArray(new_array)

    if isinstance(sitk_img, sitk.Image):
        spacing = sitk_img.GetSpacing()
        direction = sitk_img.GetDirection()

        for k in sitk_img.GetMetaDataKeys():
            return_image.SetMetaData(k, sitk_img.GetMetaData(k))

        return_image.SetSpacing(spacing)
        return_image.SetDirection(direction)
    elif isinstance(sitk_img, dict):

        return_image.SetSpacing(sitk_img['spacing'])
        return_image.SetDirection(sitk_img['direction'])
        other_keys = list(sitk_img.keys())
        other_keys.remove('spacing')
        other_keys.remove('direction')
        for k in list(other_keys):
            return_image.SetMetaData(k, sitk_img[k])
    else:
        print('Not Implemented')
    return return_image




def thresholdCT(img, thresh):
    return_img = (img>=thresh).astype(np.float32)
    return return_img


def dice_coef2(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union == 0:
        return 1
    intersection = np.sum(y_true_f * y_pred_f)
    return 2. * intersection / union


def get_dice(y1, y2, on='bone_mask', thresh=70, plot=False):
    y1_arr = sitk.GetArrayFromImage(y1)
    y2_arr = sitk.GetArrayFromImage(y2)

    if on == 'bone_mask':
        y1_arr = get_bone_mask(y1_arr, thresh)
        y2_arr = get_bone_mask(y2_arr, thresh)
    elif on == 'bone_mask_brain':
        y1_arr = thresholdCT(y1_arr, thresh)
        y2_arr = thresholdCT(y2_arr, thresh)

    dice_list = [dice_coef2(y1_arr[z, :, :], y2_arr[z, :, :]) for
                 z in range(len(y1_arr))]
    if plot == True:
        y1_view = get_sitkimage(y1, y1_arr)
        y2_view = get_sitkimage(y2, y2_arr)
        return np.mean(dice_list), dice_list, y1_view, y2_view
    else:
        return np.mean(dice_list), dice_list


def get_bone_mask(np_array, thresh=100):
    bone = ct_window(np_array, bone_window)
    return thresholdCT(bone, thresh)


def get_2d_mask(img):
    img = thresholdCT(img, thresh=0)
    img = img.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[:, 4]
    # get second largest connected componet mask
    ind = np.argsort(sizes)[-2]
    mask = (output == ind).astype('uint8')

    # find countour and fill
    contour, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
    contour_image = np.zeros_like(mask)
    for cnt in contour:
        cv2.fillPoly(contour_image, [cnt], color=255)

    # dilate countour image
    kernel = np.ones((4, 4))
    dilated  = cv2.dilate(contour_image, kernel, iterations=3)

    return dilated.astype('bool')


def get_3d_mask(image):
    bone_window = (3000, 500)
    if isinstance(image, sitk.Image):
        array = sitk.GetArrayFromImage(image)
    else:
        array = image
    bone = ct_window(array, bone_window)
    mask_3d = np.stack([get_2d_mask(bone[i, :, :])
                        for i in range(bone.shape[0])], axis=0)
    return mask_3d


def save_registration(transform, resampled, stats, output_dir, uid):
    print(f'Saving {uid}')
    sitk.WriteImage(resampled,
                    os.path.join(output_dir, f'{uid}_resampled.mha'))
    sitk.WriteTransform(transform,
                        os.path.join(output_dir, f'{uid}_transform.tfm'))
    json.dump(stats,
              open(os.path.join(output_dir, f'{uid}_stats.json'), 'w'))
