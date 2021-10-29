import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from amrit.utils.augmentations import get_augmentations , get_augmentations_plain
from amrit.utils.dimensionality import get_ImageDataset_tSNE


class imageDataset(Dataset):
    def __init__(self, dataframe, path_column='file_path', label_column='target', transform=None) -> None:
        """dataframe - pandas dataframe with 2 columns - file_path and target."""
        super().__init__()
        self.df = dataframe
        self.transform = transform
        self.labels = self.df[label_column]
        self.paths = self.df[path_column]
        self.classes = self.labels.unique()
        self.classes_dict = {key: i for i, key in enumerate(self.classes)}

        assert len(self.labels) == len(self.paths), f"{path_column} and {label_column} should have same length"
        print("data size - ", len(self.paths))
        print('classes = ', self.classes)

    def __getitem__(self, index):
        file_path = self.paths.iloc[index]
        label = self.labels.iloc[index]
        #image = Image.open(file_path).convert('RGB')

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']
            #image = self.transform(image)
            label = self.target_transformations(label)
        return image, label

    def __len__(self):
        return len(self.paths)

    def target_transformations(self, label):
        return torch.tensor(self.classes_dict.get(label))


class Classification_imageDataset(imageDataset):
    def __init__(self, dataframe, path_column='file_path', label_column='target', transform=None) -> None:
        """dataframe - pandas dataframe with 2 columns - file_path and target."""
        super().__init__()

class Segmentation_ImageDataset(Dataset):
    def __init__(self, dataframe, path_column='file_path', label_column='target', transform=None) -> None:
        """dataframe - pandas dataframe with 2 columns - file_path and target."""
        super().__init__()
        self.df = dataframe
        self.transform = transform
        self.label_path = self.df[label_column]
        self.paths = self.df[path_column]

        #Classes - Unique pixel values in masks

    def __getitem__(self, index):
        file_path = self.paths.iloc[index]
        label_path = self.label_path.iloc[index]
        #image = Image.open(file_path).convert('RGB')

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            tranformed = self.transform(image=image , mask=label)
            image = tranformed['image']
            label = tranformed['mask']

        return image, label

    def __len__(self):
        return len(self.paths)

    def target_transformations(self, label):
        return torch.tensor(self.classes_dict.get(label))



class imageDataLoader(LightningDataModule):
    def __init__(
            self,
            train_df=None,
            val_df=None,
            test_df = None,
            aug_p: float = 0.5,
            img_sz: int = 124,
            batch_size: int = 16,
            num_workers: int = 4,
            shuffle = True,
            augment = True ,
            task = 'classification', # 'segmentation'
            path_column='file_path',
            label_column='target',
    ):
        super().__init__()

        self.train_df, self.val_df, self.test_df = train_df, val_df , test_df
        self.aug_p = aug_p
        self.img_sz = img_sz
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        if augment == True:
            self.train_tfms, self.valid_tfms = get_augmentations(self.aug_p, image_size=self.img_sz)
        else:
            self.train_tfms, self.valid_tfms = get_augmentations_plain(self.aug_p, image_size=self.img_sz)
        self.path_column= path_column
        self.label_column =  label_column

        implemented_tasks = ['classification', 'segmentation']

        assert  task in implemented_tasks , f"Implemented tasks are - {implemented_tasks}"
        self.task = task

    def prepare_data(self):
        self.train_df = self.train_df  # imageDataset(train_df , transform = self.train_tfms)
        self.val_df = self.val_df  # imageDataset(valid_df, transform = self.valid_tfms)
        self.test_df = self.test_df  # imageDataset(valid_df, transform = self.valid_tfms)

    def train_dataloader(self):

        if self.task == 'classification':
            train_dataset = Classification_imageDataset(dataframe=self.train_df,path_column= self.path_column ,
                                         label_column= self.label_column, transform=self.train_tfms)
        if self.task == 'segmentation':
            train_dataset = Segmentation_ImageDataset(dataframe=self.train_df,path_column= self.path_column ,
                                         label_column= self.label_column, transform=self.train_tfms)

        return DataLoader( train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=True,)

    def test_dataloader(self):

        if self.task == 'classification':
            test_dataset = Classification_imageDataset(dataframe=self.test_df, path_column= self.path_column ,
                                     label_column= self.label_column, transform=self.valid_tfms)
        if self.task == 'segmentation':
            test_dataset = Segmentation_ImageDataset(dataframe=self.test_df, path_column= self.path_column ,
                                     label_column= self.label_column, transform=self.valid_tfms)

        return DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle,
                pin_memory=True,)

    def predict_dataloader(self):

        if self.task == 'classification':
            predict_dataset = Classification_imageDataset(dataframe=self.test_df, path_column= self.path_column ,
                                     label_column= self.label_column, transform=self.valid_tfms)
        if self.task == 'segmentation':
            predict_dataset = Segmentation_ImageDataset(dataframe=self.test_df, path_column=self.path_column,
                                           label_column=self.label_column, transform=self.valid_tfms)

        return DataLoader(
                predict_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle,
                pin_memory=True,)

    def val_dataloader(self):

        if self.task == 'classification':
            val_dataset = Classification_imageDataset(dataframe=self.valid_df, path_column=self.path_column,
                                       label_column=self.label_column, transform=self.valid_tfms)
        if self.task == 'segmentation':
            val_dataset = Segmentation_ImageDataset(dataframe=self.valid_df, path_column=self.path_column,
                                       label_column=self.label_column, transform=self.valid_tfms)

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=True,)

    def sample(self , inp, title = None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

        # # Get a batch of training data
        # inputs, classes = next(iter(train_dataloader))
        # # Make a grid from batch
        # out = torchvision.utils.make_grid(inputs)
        # imshow(out)


    def visualize(self , type = 'train' , count=9, subplot=(3, 3),  unnormalize = True ,
                  cmap = 'gist_gray', random = True, figheight = 10, figwidth = 20):
        if type == 'test':
            df = self.test_df
            transform = self.valid_tfms
        elif type == 'val':
            df = self.val_df
            transform = self.valid_tfms
        else:
            df = self.train_df
            transform = self.train_tfms

        dataset = imageDataset(dataframe= df, path_column=self.path_column,
                                   label_column=self.label_column, transform=transform)

        image_batch_visualizer2d( dataset, count= count, subplot=subplot,   unnormalize =  unnormalize ,
                                    cmap = cmap, random_img=random,
                             figheight=figheight, figwidth=figwidth)

    def get_tSNE(self, data= None,  n_components=180 , perplexity=80.0  , max_count = 2500 ):
        if data == 'test':
            df = self.test_df
        elif data == 'val':
            df = self.valid_df
        else:
            df = self.train_df
        
        if df.shape[0] > max_count:
            df = df.sample(max_count)
            print(f'Image count {df.shape[0]}, so plotting tSNE for sub sample of {max_count}')
 
        plt.figure(figsize=(15,15))
        print(f'starting tSNE for {data}')
        get_ImageDataset_tSNE(df , n_components=n_components , perplexity=perplexity ,
                              path_column = self.path_column , label_column = self.label_column)
        plt.show()


    def compare_TSNE(self , n_components=180 , perplexity=40, max_count = 500 ):

        if self.train_df.shape[0] > max_count:
            train_data = self.train_df.sample(max_count)
        else:
            train_data = self.train_df

        if self.valid_df.shape[0] > max_count:
            valid_data = self.valid_df.sample(max_count)
        else:
            valid_data = self.train_df

        if self.test_df.shape[0] > max_count:
            test_data = self.test_df.sample(max_count)
        else:
            test_data = self.train_df

        plt.figure(figsize=(20, 8))
        plt.subplot(1,3,1)
        get_ImageDataset_tSNE(train_data,  n_components=n_components , perplexity=perplexity ,
                              path_column = self.path_column , label_column = self.label_column)
        plt.title('Train')
        plt.subplot(1,3,2)
        get_ImageDataset_tSNE(valid_data,  n_components=n_components , perplexity=perplexity ,
                              path_column = self.path_column , label_column = self.label_column)
        plt.title('Valid')
        plt.subplot(1,3,3)
        get_ImageDataset_tSNE(test_data,  n_components=n_components , perplexity=perplexity,
                              path_column = self.path_column , label_column = self.label_column)
        plt.title('Test')
        plt.show()


def image_batch_visualizer2d(dataset, count=9, subplot=(3, 3),  unnormalize = True ,
                               cmap='gist_gray', random_img=True,
                             figheight=10, figwidth=20 ):
    fig = plt.figure(figsize=(figwidth, figheight))

    #dataset.df = shuffle(dataset.df)
    images_shown = 0

    while images_shown < count:
        if random_img == True:
            index = random.randint(0, dataset.df.shape[0] -1  )

        image , label = dataset[index]
        image = image.numpy().transpose((1, 2, 0))

        if unnormalize == True:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)

        plt.subplot(subplot[0], subplot[1], images_shown + 1)
        plt.tight_layout()
        plt.imshow(image , cmap=cmap)
        plt.title(label)

        images_shown += 1
    plt.show()








