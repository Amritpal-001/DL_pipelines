import json
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from .utils import get_sitkimage
import sys
import os
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import imageio
sys.path.insert(0, '/home/users/mayug.maniparambil/head_ct/hemorrhages/notebook_helper/')
from sitk_vis import myshow3d
from temp_utils import get_pred, plot_target, show_linear, thresholding
sys.path.insert(0, '/home/users/mayug.maniparambil/head_ct/hemorrhages/new/fractures-2/')
from amrit.utils.preprocessing import simple_resize_img

def _crop_inverse(pred, crop):
    size = (crop[0, 1] - crop[0, 0 ], crop[1, 1] - crop[1, 0])
#     print(size)
    resize = simple_resize_img(size[0], size[1])
    pred = pred.astype(np.float32)
    return_image = np.zeros((pred.shape[0], 512, 512))
    inlay_image = np.stack([resize(pred[i, :, :]) for i in range(pred.shape[0])])
    inlay_image = inlay_image.squeeze()
#     print('inlay_image.shape', inlay_image.shape)
    return_image[:, crop[0, 0]: crop[0, 1], crop[1, 0]: crop[1, 1]] = inlay_image
    return return_image

def cache_moving_image(sitk_dataset_orig):
    cache_dict = {}
    
    for uid in tqdm(list(sitk_dataset_orig.gt_table['StudyUID'])):
        print(uid)
        try:
            moving_image = sitk_dataset_orig.get_scan(uid)[0]
            cache_dict[uid] = get_dcm_meta_dict(moving_image)
        except Exception as e:
            print(f'skipped {uid} because of {e}')
    json.dump(cache_dict, open('./dcm_metadata_cache.json', 'w'))

def get_dcm_meta_dict(image):

    return_dict= {}
    return_dict['spacing'] = image.GetSpacing()
    return_dict['direction'] = image.GetDirection()
    for k in image.GetMetaDataKeys():
        return_dict[k] = image.GetMetaData(k)
    return return_dict


def plot_bleeds_topology(bleeds_sum, fixed_image_arr, plot=False):
    if not isinstance(fixed_image_arr, np.ndarray):
        fixed_image_arr = sitk.GetArrayFromImage(fixed_image_arr)
    bleeds_sum_max = bleeds_sum.max()

    line = cm.hot(np.linspace(0,1, 100))
    scale = np.concatenate([np.expand_dims(line, axis=0) for i in range(100)], axis=0)
    
    if plot:
        plt.figure()
        plt.imshow(scale)
        plt.xlabel(f'0 to {bleeds_sum_max}')
    
    bleeds_sum_norm = bleeds_sum / bleeds_sum_max
    
    heatmap = cm.hot(bleeds_sum_norm)
    print(heatmap.shape)
    heatmap = heatmap[:,:,:,:-1]
    print(heatmap.shape)
    fixed_image_arr = (fixed_image_arr - fixed_image_arr.min()) / ((fixed_image_arr.max() - fixed_image_arr.min()))

    display = np.zeros(fixed_image_arr.shape+(3,))
    display[:,:,:,0], display[:,:,:,1], display[:,:,:,2]  = fixed_image_arr, fixed_image_arr, fixed_image_arr
    
    if plot:
        myshow3d(sitk.GetImageFromArray((display + heatmap)/2))
    return (display + heatmap)/2, bleeds_sum_norm


def get_bleeds_sum(consider_table, sitk_dataset, fixed_image, pred_folder, reg_folder, metadata_cache, chronic=False):
    # sitk dataset has to return sitk image 512*512 with all dicom metadata --> nii gz
    assert sum(list(consider_table['final_dsc']>0.5)) == len(consider_table)
    fixed_image_arr = sitk.GetArrayFromImage(fixed_image)
    bleeds_sum = np.zeros_like(fixed_image_arr)
    ctr = 0
    for uid in tqdm(list(consider_table['StudyUID'])):
        
        try:
            # pred = get_pred(pred_folder, uid, mask_number=0, chronic=chronic)
            # crop = sitk_dataset.get_crop_uid(uid)
            # pred_invcrop = _crop_inverse(pred, crop)
            # pred_invcrop = pred_invcrop.astype(np.float32)

            # try:
            #     moving_image = metadata_cache[uid]
            # except KeyError:
            #     moving_image = sitk_dataset.get_scan(uid)[0]
            # pred_invcrop_image = get_sitkimage(moving_image, pred_invcrop)

            # # get trasnform and apply transform

            # transform = sitk.ReadTransform(os.path.join(reg_folder, f'{uid}_transform.tfm'))
            # pred_resampled_image = sitk.Resample(pred_invcrop_image, fixed_image,
            #                              transform, sitk.sitkNearestNeighbor,
            #                              0.0, pred_invcrop_image.GetPixelID())

            pred_resampled_image = get_pred_resampled(pred_folder,
                                                      reg_folder,
                                                      uid,
                                                      fixed_image,
                                                      mask_number=0,
                                                      chronic=chronic,
                                                      metadata_cache=metadata_cache,
                                                      sitk_dataset=sitk_dataset)
            bleeds_sum = bleeds_sum + sitk.GetArrayFromImage(pred_resampled_image)
            ctr = ctr + 1
        except Exception as e:
            print(f'{uid} not in bleeds sum because of {e}')
    return bleeds_sum, ctr


def get_pred_resampled(pred_folder, reg_folder, uid,
                       fixed_image,
                       mask_number=0, chronic=False,
                       metadata_cache=None, sitk_dataset=None):
    pred = get_pred(pred_folder,
                    uid, mask_number=0, 
                    chronic=chronic)
    crop = sitk_dataset.get_crop_uid(uid)
    pred_invcrop = _crop_inverse(pred, crop)
    pred_invcrop = pred_invcrop.astype(np.float32)

    try:
        moving_image = metadata_cache[uid]
    except KeyError:
        moving_image = sitk_dataset.get_scan(uid)[0]
    pred_invcrop_image = get_sitkimage(moving_image, pred_invcrop)

    # get trasnform and apply transform

    transform = sitk.ReadTransform(os.path.join(reg_folder,
                                                f'{uid}_transform.tfm'))

    pred_resampled_image = sitk.Resample(pred_invcrop_image,
                                         fixed_image,
                                         transform,
                                         sitk.sitkNearestNeighbor,
                                         0.0,
                                         pred_invcrop_image.GetPixelID())

    return pred_resampled_image