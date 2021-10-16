import torch
import random
import warnings
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from amrit.explain.explain_helpers import load_image
from amrit.dataloaders.tabular import tabularData
from amrit.dataloaders.image import imageDataset, imageDataLoader
from amrit.models.tabular import tabularmodel
from amrit.models.cnnModel import ImageModel
from amrit.predictions.analyze_preds import prediction_plotter
from amrit.utils.dataModifier import set_seed , get_labels_df_from_folder, trainTestSplit
from amrit.utils.augmentations import get_augmentations
from amrit.utils.dataModifier import trainTestSplit
from amrit.utils.filefolder import get_random_Imagepath , get_random_list_Imagepath
from amrit.explain.explain_cnn import get_guided_GradCAM , plot_gradcam_comparison, plot_in_grid , plot_gradcam_single_image
from PIL import Image
from amrit.explain.feature_maps import create_forward_hook , create_backward_hook , visualize_feature_maps


warnings.filterwarnings("ignore")

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import numpy as np
from glob import glob
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def visualize_scatter(data_2d, label_ids, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.grid()

    nb_classes = len(np.unique(label_ids))

    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color=plt.cm.Set1(label_id / float(nb_classes)),
                    #linewidth='1',
                    alpha=0.8,
                    label=id_to_label_dict[label_id])
    plt.legend(loc='best')
    plt.show()



def get_folder_numpy(folder_path):
    images = []
    labels = []

    for class_folder_name in os.listdir(folder_path):
        print(class_folder_name)
        class_folder_path = os.path.join(folder_path, class_folder_name)
        for image_path in glob(os.path.join(class_folder_path, "*.jpg")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            image = cv2.resize(image, (150, 150))
            #image = segment_plant(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (45, 45))

            image = image.flatten()

            images.append(image)
            labels.append(class_folder_name)

    images = np.array(images)
    labels = np.array(labels)

    return images , labels


def get_dataframme_numpy(df):
    images = []
    labels = []

    for index in range(len(df['file_path'])):
            image_path = df['file_path'].iloc[index]
            label = df['target'].iloc[index]
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            image = cv2.resize(image, (150, 150))
            #image = segment_plant(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (45, 45))

            image = image.flatten()

            images.append(image)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images , labels

# Create df
train_path = "../data/cat_dog/Train"
train_path = "../data/medical"
train_df = get_labels_df_from_folder(train_path , extension='.jpg')
train_df , valid_df = trainTestSplit(df=train_df, mode='random', test_size = 0.5, split_random_seed= 42)
test_df , valid_df = trainTestSplit(df=valid_df, mode='random', test_size = 0.5, split_random_seed= 42)

# Datalaoder from df
loader = imageDataLoader(train_df, valid_df, test_df = test_df , aug_p = 0.5,
                         img_sz = 128, batch_size= 32, num_workers = 2, )

loader.compare_TSNE()