

import matplotlib.pyplot as plt

from .explain_helpers import load_image, show_mask, show_mask_on_image, cut_image_with_mask
from .saliency.grad_cam import GradCam
from .saliency.guided_backprop import GuidedBackprop


def get_GradCAM(image, model, image_class_id = 0):
    grad_cam = GradCam(model)
    image_region = grad_cam.get_mask(image_tensor=image, target_class=image_class_id)
    grad_cam.remove_hooks()
    return(image_region)

def get_guided_GradCAM(image, model, image_class_id = 0):
    guided_backprop = GuidedBackprop(model)
    image_mask = guided_backprop.get_mask(image_tensor=image, target_class=image_class_id)
    guided_backprop.remove_hooks()

    grad_cam = GradCam(model)
    image_region = grad_cam.get_mask(image_tensor=image, target_class=image_class_id)
    grad_cam.remove_hooks()

    # print(image.shape)
    # print(image_mask.shape)
    # print(image_region.shape)

    # 'Guided Grad-CAM' is a combination of the 'Guided Backprop' and 'Grad-CAM' method.
    guided_grad_cam_image = guided_backprop.apply_region(image_mask, image_region)
    return(guided_grad_cam_image , image_mask , image_region)


def plot_gradcam_comparison(image_path , model , image_class_id= 0 ):
    image = load_image(image_path, size=299)
    guided_grad_cam_image, image_mask , gradcam = get_guided_GradCAM(image, model, image_class_id=0)

    figure, axes = plt.subplots(1, 2, figsize=(8, 8), tight_layout=True)
    # show_mask_on_image(image_path= image_path , mask=gradcam, title='Grad-CAM',axis=axes[0])
    # show_mask(guided_grad_cam_image, title='Guided Grad-CAM: Boxer', axis=axes[1])
    show_mask_on_image(image_path=image_path, mask=gradcam, title='GradCAM', axis=axes[0])
    show_mask(guided_grad_cam_image, title='Guided GradCAM', axis=axes[1])
    plt.show()


def plot_gradcam_batch_comparison(image_path , model , image_class_id= 0 ):
    image = load_image(image_path, size=299)
    guided_grad_cam_image, image_mask , gradcam = get_guided_GradCAM(image, model, image_class_id=0)

    figure, axes = plt.subplots(1, 2, figsize=(8, 8), tight_layout=True)
    # show_mask_on_image(image_path= image_path , mask=gradcam, title='Grad-CAM',axis=axes[0])
    # show_mask(guided_grad_cam_image, title='Guided Grad-CAM: Boxer', axis=axes[1])
    show_mask_on_image(image_path=image_path, mask=gradcam, title='GradCAM', axis=axes[0])
    show_mask(guided_grad_cam_image, title='Guided GradCAM', axis=axes[1])
    #plt.show()


def plot_gradcam_single_image(image_path , model , label = None,  cut_with_mask = True , image_size = 224 , image_class_id= 0 , percentile=90 ):
    image = load_image(image_path, size=image_size)
    guided_grad_cam_image, image_mask , gradcam = get_guided_GradCAM(image, model, image_class_id= image_class_id)
    if cut_with_mask == True:
        cut_image_with_mask(image_path=image_path, mask=gradcam, title= label , percentile=percentile )
    else:
        show_mask_on_image(image_path=image_path, mask=gradcam, title= label )


#plt.show()

def plot_in_grid(image_list, function, model, label_list , grid_size = 3 , cut_with_mask = True , image_class_id= 0 , percentile=90 ):
    figure = plt.figure(figsize=(20,20))
    index = 1
    for index in range(1, len(image_list)):
        plt.subplot(grid_size,grid_size,index)
        function(image_list[index-1], model , label_list[index-1] , cut_with_mask = cut_with_mask,
                    image_class_id= image_class_id, percentile = percentile)
        index += 1
    plt.show()

'''from saliency.utils import load_image, make_grayscale, make_black_white, show_mask, show_mask_on_image, cut_image_with_mask
from saliency.vanilla_gradient import VanillaGradient
from saliency.integrated_gradients import IntegratedGradients


def get_vanillagradients(image ,model):
    # Construct a saliency object and compute the saliency mask.
    vanilla_gradient = VanillaGradient(model)
    rgb_mask = vanilla_gradient.get_mask(image_tensor=image)
    # Make a black and white variant of the computed saliency mask.

def get_guided_Backprop(image, model):
    guided_backprop = GuidedBackprop(model)
    rgb_mask = guided_backprop.get_mask(image_tensor=doberman)
    # Vanilla Gradient, Guided Backpropagation and Integrated Gradients also implement a SmmothGrad variant of their saliency mask.
    smooth_rgb_mask = guided_backprop.get_smoothed_mask(image_tensor=doberman, samples=50)
    # ReLUs are modified in PyTorch using hooks which we need to remove after we are done.
    guided_backprop.remove_hooks()
    smooth_rgb_mask = make_black_white(smooth_rgb_mask)

    return(smooth_rgb_mask)

def get_integraded_gradients(image, model):
    integrated_gradients = IntegratedGradients(model)
    # 'get_mask' method. By passing e.g. 'np.abs' the visual result can be improved.
    abs_rgb_mask = integrated_gradients.get_mask(image_tensor=doberman, process=np.abs)
    abs_rgb_mask = make_black_white(abs_rgb_mask)
    return(abs_rgb_mask)
'''
