
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io
from skimage.transform import rescale
from sklearn.utils import shuffle
from torch.utils.data import Dataset


class image_2D_dataset(Dataset):
    def __init__(self, df ,  image_path_col = 'filepath' , label_col = 'label' ,root_dir= None, transform=None, 
                   sub_labels :list = None ,do_shuffle = True, rescale = False):

        if root_dir == None:
            root_dir = './'

        self.df = df 
        print("Sub_labels avialable - " , self.df[label_col].unique())
        
        self.image_path_col = image_path_col
        
        if sub_labels == None and do_shuffle == True :
            self.df = shuffle(self.df)
        else:
            #assert sub_labels in self.df[label_col].unique() , f" choose  an option from {self.df[label_col].unique()}"
            for sub_label in sub_labels:
                    self.df = self.df[self.df[label_col] == sub_label]
        if do_shuffle == True :
                self.df = shuffle(self.df)
        print(self.df.shape)
        
        self.df = self.df.reset_index()
        self.root_dir = root_dir
        self.transform = transform
        self.label_col = label_col
        self.rescale = rescale 
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,  self.df[self.image_path_col][idx])
        image = io.imread(img_name)
        
        if self.rescale == True:
            image = rescale(image, 0.10, anti_aliasing=False)
        label = self.df[self.label_col][idx]
        label = np.array([label])
        
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label}
        return sample


def image_visualizer2d(dataset , rand = True, cmap = 'gist_gray'):
    fig = plt.figure(figsize = (16,16))

    if rand == True:
        i = random.randint(0,len(dataset))
    else:
        i = 0
        
    sample = dataset[i]
    plt.tight_layout()
    plt.imshow(sample['image'], cmap = cmap)
    plt.title(sample['label'])
    plt.show()
    #input("Good(g) or Bad(b)")

def image_batch_visualizer2d(dataset , count = 9 , subplot = (3,3), cmap = 'gist_gray' , random = True,
              figheight = 10 , figwidth = 20):

    
    fig = plt.figure(figsize = (figwidth, figheight))
    
    if random == True:
        dataset.df = shuffle(dataset.df)
    for i in range(len(dataset)):
        sample = dataset[i]
        plt.subplot(subplot[0] , subplot[1] , i + 1)
        plt.tight_layout()
        plt.imshow(sample['image'], cmap = cmap)
        plt.title(sample['label'])
           
        if i == count - 1:
            break
    plt.show()
            

    
class Multilabel_ImageDataset(Dataset):
    def __init__(self, df , label_col , root_dir, transform=None, 
                 sub_label_col = None, sub_labels :list = None,
                do_shuffle = True):

        self.df = df 
        print("Sub_labels avialable - " , self.df[label_col].unique())

        if sub_labels == None and do_shuffle == True :
            self.df = shuffle(self.df)
        else:
            assert sub_labels in self.df[label_col].unique() , f" choose  an option from {self.df[label_col].unique()}"
            if len(sub_labels) == 1:
                for sub_label in sub_labels:
                    self.df = self.df[self.df[label_col] == sub_label]
            else:
                for index in range(len(sub_labels)):
                    self.df = self.df[self.df[sub_label_col[index]] == sub_labels[index]]
                if do_shuffle == True :
                    self.df = shuffle(self.df)
            print(self.df.shape)
        
        self.df = self.df.reset_index()
        self.root_dir = root_dir
        self.transform = transform
        self.label_col = label_col
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,  self.df['file_name'][idx][:-4] + '.png')
        image = io.imread(img_name)
        image = rescale(image, 0.10, anti_aliasing=False)
        label = self.df[self.label_col][idx]
        label = np.array([label])
        
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label}
        return sample