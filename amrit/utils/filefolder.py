
import glob
import os.path
import random

import cv2
import numpy as np
import pandas as pd


def get_file_list(folder_path : str = None ,extension = '.dcm', recursive = False):
    assert folder_path != None , "please provide a folder path"
    if recursive == False:
        file_list = glob.glob(folder_path + './*' +extension , recursive=True)
    else:
        file_list = glob.glob(folder_path + '/**/*' +  extension , recursive=True)
    file_list = sorted(file_list)
    
    return(file_list)


def get_folder_dataframe(folder_path: str = None, extension='.dcm', recursive=False):
    file_list = get_file_list(folder_path = folder_path ,extension = extension , recursive = recursive)

    df = pd.DataFrame(columns=['file_name', 'metadata'])

    for x in range(len(file_list)):
        file_name = file_list[x]
        df = df.append(pd.Series([os.path.basename(file_name),  os.path.dirname(file_name) ], index=df.columns), ignore_index=True)
    return (df)


def get_random_Imagepath(train_df):
    return train_df['file_path'][random.randint(0,train_df.shape[0])]

def get_random_list_Imagepath(train_df, count = 9):
    pathlist = []
    labellist = []
    while len(pathlist) < count:
        random_num = random.randint(0, train_df.shape[0])
        pathlist.append(train_df['file_path'].iloc[random_num]  )
        labellist.append(train_df['target'].iloc[random_num] )
    return pathlist , labellist



def get_Flatted_Numpy_Images_from_Folder(folder_path):
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


def get_Flatted_Numpy_Images_from_DataFrame(df , path_column = 'file_path' , label_column = 'target'):
    images = []
    labels = []

    for index in range(len(df[path_column])):
            image_path = df[path_column].iloc[index]
            label = df[label_column].iloc[index]
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
