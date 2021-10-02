# helper functions for grid parcellation of template and creation of bleed coverage csv.

from .visualization import get_pred_resampled
import SimpleITK as sitk
import sys
sys.path.insert(0, '/home/users/mayug.maniparambil/head_ct/hemorrhages/notebook_helper/')
from temp_utils import get_pred, thresholding
sys.path.insert(0, '/home/users/mayug.maniparambil/head_ct/hemorrhages/new/fractures-2/')
from src.validation.misc_utils import SITKDatasetSqlite

def overlap(bleed, label, mean=True):
    assert bleed.shape == label.shape
    all_labels = np.unique(label)
    overlap = {}
    for a in all_labels:
        label_map = (label==a)
        if mean:
            overlap[a] = (label_map & bleed).sum() / label_map.sum()
        else:
            overlap[a] = (label_map & bleed).sum()
    return overlap


def overlap_3d(registered_bleed, label):
    slice_overlap = []
    for i in range(label.shape[0]):
        slice_overlap.append(overlap(registered_bleed[i, :, :], label[i, :, :]))
    return slice_overlap



def add_to_df(slice_overlaps, df, uid):
    print(uid)
    for i, s in enumerate(slice_overlaps):
        for k,o in s.items():
#             print('k', k)
#             print('o', o)
            label = int(k)
#             print('label ', label)
            column = f'{i}_{label}'
#             print('column', column)
            df.loc[uid, column] = o


def create_csv(reg_table, pred_folder, uid, fixed_image, metadata_cache,
               sitk_dataset, output_file_name):
    i = 0
    df = pd.DataFrame(columns=[f'{int(i)}_{int(l)}' for i in range(len(label_arr)) for l in range(1, int(label_arr[i,:,:].max()) + 1) ])

    for uid in tqdm(list(reg_table['StudyUID'])):
        try:
            registered_bleed = get_pred_resampled(pred_folder, reg_folder,
                                                  uid, fixed_image, metadata_cache=metadata_cache,
                                                  sitk_dataset=sitk_dataset)
            registered_bleed_arr = sitk.GetArrayFromImage(registered_bleed)
            registered_bleed_arr = thresholding(registered_bleed_arr, 0.5).astype(np.uint8)
            slice_overlaps = overlap_3d(registered_bleed_arr, label_arr)
            add_to_df(slice_overlaps, df, uid)    
            df.to_csv(output_file_name)
        except Exception as e:
            print(f"Skipping uid = {uid} because of {e}")
        i = i + 1
    return df

# Use label_single_dilated_eroded and label_arr(grid) to calculate scaling coefficients for each
# periphery grid in each slice 

def return_periphery_scaling_factors(label_map, label_map_grid):
    # label_map corresponds to outline of brain
    # label_map_grid corresponds to 
    slice_overlaps = overlap_3d(label_map, label_map_grid)
    return_dict = {}
    for i, s in enumerate(slice_overlaps):
        for k, coeff in s.items():
            if coeff !=0 and coeff!=1:
                return_dict[f'{i}_{int(k)}'] = coeff
    return return_dict


def normalize_periphery(table_csv_path, grid_label_arr):
    parcellation_table = pd.read_csv(table_csv_path)
    skull_map = get_skull_map()
    periphery_scaling_factors = return_periphery_scaling_factors(get_skull_map,
                                                                 grid_label_arr)

    for k, coeff in periphery_scaling_factors.items():
        parcellation_table[k] = parcellation_table[k] / periphery_scaling_factors[k]
    
    # remove all subjects where any coverage is greater than 1

    for k in periphery_scaling_factors.keys():
        parcellation_table = parcellation_table[parcellation_table[k]<1]
    print('saving normalized periphery')
    parcellation_table.to_csv(table_csv_path[:-len('.csv')+'_normalized.csv'])



 def get_skull_map():
    template_folder = '/home/users/mayug.maniparambil/head_ct/registration/registration_framework/templates/'
    label_coarse =  sitk.ReadImage(os.path.join(template_folder, 'rire_template_structures.nrrd'))
    label_coarse_arr = sitk.GetArrayFromImage(label_coarse)

    label_single = label_coarse_arr / label_coarse_arr.max()
    label_single = (label_single>0.0001).astype(np.uint8)
   
   # dilate and erode

    kernel = np.ones((10,10),np.uint8)
    label_single_dilated = np.stack([cv2.dilate(label_single[i, :, :], kernel, iterations=2) for i in range(len(label_single))], )
    kernel = np.ones((3,3),np.uint8)
    label_single_dilated_eroded = np.stack([cv2.erode(label_single_dilated[i, :, :], kernel, iterations=2) for i in range(len(label_single_dilated))], )

    return label_single_dilated_eroded




if __name__=='main':
    template_folder = '/home/users/mayug.maniparambil/head_ct/registration/registration_framework/templates/'
    template = os.path.join(template_folder, 'rire_train_01_ct-template.mha')
    template_grid = './parcellations_grid_centered.nrrd'
    pred_folder = '/home/users/mayug.maniparambil/head_ct/hemorrhages/new/fractures-2/preds/hmgs_224_encoder-resnet18_decoder-gatedsseunet-pretrained_loss-mixedfocal-g2w1m1_dataset-acute-fine_sqlite-05May_twomask-total_sampling0.25_lr10_crop_tes_any_crop_corrected/'
    reg_folder = '/home/users/mayug.maniparambil/head_ct/registration/registration_framework/registrations/run1/'
    reg_table = pd.read_csv('/home/users/mayug.maniparambil/head_ct/hemorrhages/segmentation_study/notebooks/nlp_test_any_run1_dsc.csv')

    label = sitk.ReadImage(template_grid)
    label_arr = sitk.GetArrayFromImage(label)

    fixed_image = sitk.ReadImage(template, sitk.sitkFloat32)
    fixed_image_arr = sitk.GetArrayFromImage(fixed_image)

    nlp_test_any = pd.read_csv('/home/users/mayug.maniparambil/head_ct/hemorrhages/segmentation_study/notebooks/nlp_test_any_crop.csv')
    sitk_dataset_orig = SITKDatasetSqlite(nlp_test_any, nii_gz=True, img_side=512,
                                 f_root_path='/data_nas2/processed/HeadCT/sampling/ct_batches_all/', return_type='not array')
    metadata_cache =  json.load(open('/home/users/mayug.maniparambil/head_ct/hemorrhages/segmentation_study/notebooks/dcm_metadata_cache.json'))

    output_file_name = 'subparcellation_bleed_coverage'
    df = create_csv(reg_table, pred_folder, uid, fixed_image, metadata_cache,
               sitk_dataset_orig, output_file_name)

    parcellation_table = pd.merge(reg_table, df, how='inner', on=['StudyUID'])

    dcm_table = pd.read_csv('/home/users/mayug.maniparambil/head_ct/hemorrhages/new/fractures-2/src/validation/features_table_nlp_test_any_crop.csv')

    parcellation_table = pd.merge(parcellation_table, dcm_table)

    # now normalize all periphery grids in the csv
    normalize_periphery(parcellation_table, label_arr)


    