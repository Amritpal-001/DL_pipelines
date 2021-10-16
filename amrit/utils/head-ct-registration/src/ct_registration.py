import SimpleITK as sitk
import numpy as np
from .utils import ct_window, get_sitkimage, get_3d_mask, get_dice

bone_window = (3000, 500)


def do_registration(fixed_image, moving_image, initial_transform,
                    loss='Matt'):
    rm = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    if loss == 'Matt':
        rm.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif loss == 'correlation':
        rm.SetMetricAsCorrelation()
    rm.SetMetricSamplingStrategy(rm.RANDOM)
    rm.SetMetricSamplingPercentage(0.01)

    rm.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    rm.SetOptimizerAsGradientDescent(learningRate=1.0,
                                     numberOfIterations=100,
                                     convergenceMinimumValue=1e-6,
                                     convergenceWindowSize=10)
    rm.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    rm.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    rm.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    rm.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    rm.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    # rm.AddCommand(sitk.sitkStartEvent, start_plot)
    # rm.AddCommand(sitk.sitkEndEvent, end_plot)
    # rm.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
    # rm.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(rm))

    final_transform = rm.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                 sitk.Cast(moving_image, sitk.sitkFloat32))
    return final_transform


def register_ct(fixed_image, moving_image, loss='Matt'):
    stats = {}
    
    moving_image_arr = sitk.GetArrayFromImage(moving_image)
    fixed_image_arr = sitk.GetArrayFromImage(fixed_image)
    # remove head rest
    head_mask_arr = get_3d_mask(moving_image)
    moving_image_arr[~head_mask_arr] = moving_image_arr.min()
    # bone window
    moving_image_arr = ct_window(moving_image_arr, bone_window)
    fixed_image_arr = ct_window(fixed_image_arr, bone_window)

    moving_image = get_sitkimage(moving_image, moving_image_arr)
    fixed_image = get_sitkimage(fixed_image, fixed_image_arr)
    
    # initial transform
    initial_transform = _get_initial_transform(fixed_image, moving_image)
    moving_resampled = sitk.Resample(moving_image, fixed_image,
                                     initial_transform, sitk.sitkLinear,
                                     0.0, moving_image.GetPixelID())

    dsc_mean, dsc_list = get_dice(fixed_image,
                                  moving_resampled,
                                  on='bone_mask',
                                  thresh=70)
    stats['init_dsc_mean'] = dsc_mean
    stats['init_dsc_slice'] = dsc_list
    # registration
    final_transform = do_registration(fixed_image, moving_image,
                                      initial_transform, loss)
    moving_resampled = sitk.Resample(moving_image, fixed_image,
                                     final_transform, sitk.sitkLinear,
                                     0.0, moving_image.GetPixelID())

    dsc_mean, dsc_list = get_dice(fixed_image, moving_resampled,
                                  on='bone_mask', thresh=70)
    stats['final_dsc_mean'] = dsc_mean
    stats['final_dsc_slice'] = dsc_list

    return final_transform, moving_resampled, stats


def _get_initial_transform(fixed_image, moving_image):
    affine_center = (20, 20, 20)
    affine_translation = (5, 6, 7)
    affine_matrix = np.identity(3).ravel()
    aff = sitk.AffineTransform(affine_matrix,
                               affine_translation,
                               affine_center)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        aff,
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    return initial_transform
