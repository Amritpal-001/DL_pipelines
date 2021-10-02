import os
import sys
import pandas as pd
from src_temp.ct_registration import register_ct
from src_temp.utils import save_registration
import SimpleITK as sitk
from tqdm import tqdm
import logging

sys.path.insert(0, '/home/users/mayug.maniparambil/head_ct/hemorrhages/new/fractures-2/')
from src.validation.misc_utils import SITKDatasetSqlite


fixed_image = sitk.ReadImage('./templates/rire_train_01_ct-template.mha')
# check template; Use rire template
# fixed_image = sitk.ReadImage('./templates/clinical_high_res_template_rotated_subsampled.mha')

print('Size of fixed image', fixed_image.GetSize())
input_csv = sys.argv[1]
output_dir = sys.argv[2]
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

logging.basicConfig(filename=os.path.join(output_dir, 'run.log'),
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


features_table = pd.read_csv(input_csv)

f_root_path = '/data_nas2/processed/HeadCT/sampling/ct_batches_all/'

sitk_dataset = SITKDatasetSqlite(features_table, nii_gz=True, img_side=512,
                                 f_root_path=f_root_path,
                                return_type='sitk')

for i in tqdm(range(len(features_table))):
    row = features_table.iloc[i]
    uid = row['StudyUID']
    try:
        moving_image = sitk_dataset.get_scan(uid)[0]
        transform, resampled, stats = register_ct(fixed_image,
                                                  moving_image,
                                                  loss='Matt')
        save_registration(transform, resampled, stats, output_dir, uid)
    except Exception as e:
        print(e)
        logging.error(f'Could not run registration for {uid}',
                      exc_info=True)

