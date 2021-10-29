import logging

import SimpleITK as sitk
import numpy as np
import open3d as o3d
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def get_point_cloud(sitk_image, threshold=None, percentile=96, sampling_perc=10):
    ## Make np image
    np_image = sitk.GetArrayFromImage(sitk_image)

    ## Decide threshold
    if not threshold:
        threshold = np.percentile(np_image, percentile)

    ## Get indices of thresholded points in np
    idx = np.where((np_image > threshold) & (np_image < 1900))

    ## Random sampling
    nums = np.arange(0, len(idx[0]))
    num_samples = (sampling_perc * len(idx[0])) // 100
    indices = np.random.choice(nums, num_samples, replace=False)

    ## Create point coordinates
    point_cloud = np.zeros((len(indices), 3))
    for i, ind in enumerate(indices):
        loc = sitk_image.TransformIndexToPhysicalPoint(
            (int(idx[2][ind]), int(idx[1][ind]), int(idx[0][ind]))
        )
        point_cloud[i] = loc

    ## Log
    logger.debug("Full Cloud: {}. Resampled: {}".format(len(idx[0]), len(point_cloud)))

    ## Create point cloud class open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    return pcd


def open3d_to_sitk_transform(open_t):
    ## Translation
    offset = (-1 * open_t[0][3], -1 * open_t[1][3], -1 * open_t[2][3])
    translation = sitk.TranslationTransform(3, offset)

    ## Rotation
    rotation = sitk.VersorTransform()
    rotation.SetMatrix(np.concatenate((open_t[:3, 0], open_t[:3, 1], open_t[:3, 2])))

    ## Return composite
    # works only for sitk>2
    # final_transform = sitk.CompositeTransform([rotation, translation])
    final_transform = sitk.Transform()
    final_transform.AddTransform(rotation)
    final_transform.AddTransform(translation)
    return final_transform


def preprocess_point_cloud(pcd, voxel_size):
    ## Downsample the cloud with voxels of given size
    pcd_down = pcd.voxel_down_sample(voxel_size)

    ## Calculate normals for each voxel, for later use
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    ## Use normals to calculate features required for the RANSAC algo
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    ransac_n = 3
    convergence = o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        convergence,
    )

    return result


def run_pcd_registration(ref_pcd, bad_image, voxel_size=10, finetune=True):
    ## 3D Point cloud
    bad_pcd = get_point_cloud(bad_image, threshold=500)

    ## Find required parameters for RANSAC and downsample
    ref_down, ref_fpfh = preprocess_point_cloud(ref_pcd, voxel_size)
    bad_down, bad_fpfh = preprocess_point_cloud(bad_pcd, voxel_size)

    ## Run RANSAC to find initial transform
    result = execute_global_registration(
        bad_down, ref_down, bad_fpfh, ref_fpfh, voxel_size
    )

    if finetune:
        ## Run ICP to fine-tune reg
        distance_threshold = voxel_size * 0.4
        result = o3d.pipelines.registration.registration_icp(
            bad_down,
            ref_down,
            distance_threshold,
            result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )

    return result


def apply_transformation(ref_image, bad_image, transformation):
    return sitk.Resample(bad_image, ref_image, 
                         open3d_to_sitk_transform(transformation))


def motion_correction(sitk_img_series):
    ref_img = sitk_img_series[0]
    ref_pcd = get_point_cloud(ref_img, threshold=500)
    
    sitk_img_series_corrected = [ref_img]

    for idx in tqdm(range(1, len(sitk_img_series)), desc='Motion correction'):
        reg = run_pcd_registration(ref_pcd, sitk_img_series[idx])
        correct_img = apply_transformation(ref_img, sitk_img_series[idx], reg.transformation)
        sitk_img_series_corrected.append(correct_img)
        
    return sitk_img_series_corrected
