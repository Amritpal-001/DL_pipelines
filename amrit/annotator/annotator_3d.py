import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import clear_output
from skimage import io
from sklearn.utils import shuffle
from torch.utils.data import Dataset


class Annotator_2D_Dataset(Dataset):
    def __init__(self, df, image_path_col = 'filepath' , label_col='label',
                 root_dir = None,  do_rescale = False,
                transform=None, do_shuffle = True, 
                  annotation_column = 'scan_quality_amrit'):

        print("Sub_labels avialable - " , df[label_col].unique())

        if root_dir == None:
            root_dir = './'

        df['Index'] = df.index
        try:
            df[annotation_column]
            print(annotation_column, 'is present df provided'  )
        except:
            df[annotation_column] = np.nan
            
        if do_shuffle == True :
            df = shuffle(df)
            print(df.shape)

        self.df = df 
        self.last_annotation = 0
        self.image_path_col = image_path_col
        #self.df = self.df.reset_index()
        self.root_dir = root_dir
        self.transform = transform
        self.label_col = label_col
        self.annotation_column = annotation_column
        self.show_subset = False
        self.rescale = do_rescale
        
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,  self.df[self.image_path_col][idx] )
        image = io.imread(img_name)
        if self.rescale == True:
            image = rescale(image, 0.10, anti_aliasing=False)
        label = self.df[self.label_col][idx]
        label = np.array([label])
        
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label, 'idx' :idx}
        return sample

    def get_subset(self , label_col , sub_labels :list = None,
                exclude_tag : list = None , include_tag = None):

        sub_df = self.df

        if exclude_tag != None:
            for tag in exclude_tag:
                sub_df = sub_df[sub_df[self.annotation_column] != tag]
            print("Sub_labels after exclusion - " , sub_df[label_col].unique())
        
        if include_tag != None:
            sub_df = sub_df[sub_df[self.annotation_column] == include_tag]
            print("Sub_labels after inclusion - " , sub_df.shape)

        if sub_labels != None:
            #assert sub_labels in self.df[label_col].unique() , f" choose  an option from {self.df[label_col].unique()}"
            for sub_label in sub_labels:
                    sub_df = sub_df[sub_df[label_col] == sub_label]

        self.sub_df = sub_df
        self.show_subset = True

    def merge_subset_annon(self,  sub_df):

        self.df = pd.merge(self.df , sub_df , how = 'right' , on = 'Index')
        
    
    def remove_subset(self):
        self.sub_df = None
        self.show_subset = False


def start_annotator_2D(dataset:Annotator_2D_Dataset , cmap = 'gist_gray' ,
                 labels_dic = None , start_point = None):
    
    changes_added = []

    if dataset.show_subset == True:
        df = dataset.sub_df
    else:
        df = dataset.df


    annotation_column = dataset.annotation_column
    
    if start_point == None:
        if dataset.last_annotation != 0:
            start_point = dataset.last_annotation
        else:
            start_point = 0
    else:
        start_point = start_point
        
    if labels_dic == None:
        labels_dic = {'g' : 'good',
            'b' : 'borderline',
            'p' : 'poor',
            'r' : 'review',}
    
    for current_index in range(start_point , len(dataset)):
        print("Scan index" , current_index , '/' , dataset.df.shape[0])
        fig = plt.figure(figsize = (8,8))
        plt.tight_layout()
        sample = dataset[current_index]
        plt.imshow(sample['image'], cmap = cmap)
        plt.title(sample['label'])
        plt.show()

        val = input("Good(g) or Poor(p) or boderline(b) , s - to save progress and quit")

        if val in labels_dic.keys():
            changes_added.append([sample['idx'] , df[annotation_column][sample['idx']] , labels_dic[val] ])
            df[annotation_column][sample['idx']]  = labels_dic[val]
        elif val == "s":
            print("saving and exiting")
            break
        clear_output(wait=True)
        dataset.last_annotation = current_index
    
    if dataset.show_subset == True:
        dataset.merge_subset_annon(df)
    else:
        dataset.df = df
    print(changes_added)


    return(dataset)
    