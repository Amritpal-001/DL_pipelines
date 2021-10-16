import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import LightningDataModule

from amrit.utils.augmentations import get_augmentations
from amrit.utils.dimensionality import  get_ImageDataset_tSNE

class imageDataset(Dataset):
    def __init__(self, dataframe, path_column='file_path', label_column='target', transform=None) -> None:
        """
        dataframe - pandas dataframe with 2 columns - file_path and target.
        """
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
        image = Image.open(file_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            label = self.target_transformations(label)
        return image, label

    def __len__(self):
        return len(self.paths)

    def target_transformations(self, label):
        return torch.tensor(self.classes_dict.get(label))


class imageDataLoader(LightningDataModule):
    def __init__(
            self,
            train_df=None,
            valid_df=None,
            test_df = None,
            aug_p: float = 0.5,
            img_sz: int = 124,
            batch_size: int = 16,
            num_workers: int = 4,
            shuffle = True,
        path_column='file_path',
        label_column='target',
    ):
        super().__init__()

        self.train_df, self.valid_df, self.test_df = train_df, valid_df , test_df
        self.aug_p = aug_p
        self.img_sz = img_sz
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.train_tfms, self.valid_tfms = get_augmentations(self.aug_p, image_size=self.img_sz)
        self.path_column= path_column
        self.label_column =  label_column

    def prepare_data(self):
        self.train_df = self.train_df  # imageDataset(train_df , transform = self.train_tfms)
        self.valid_df = self.valid_df  # imageDataset(valid_df, transform = self.valid_tfms)
        self.test_df = self.test_df  # imageDataset(valid_df, transform = self.valid_tfms)

    def train_dataloader(self):
        train_dataset = imageDataset(dataframe=self.train_df,path_column= self.path_column ,
                                     label_column= self.label_column, transform=self.train_tfms)

        return DataLoader( train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=True,)

    def test_dataloader(self):
            test_dataset = imageDataset(dataframe=self.test_df, path_column= self.path_column ,
                                     label_column= self.label_column, transform=self.valid_tfms)

            return DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle,
                pin_memory=True,)

    def predict_dataloader(self):
            predict_dataset = imageDataset(dataframe=self.test_df, path_column= self.path_column ,
                                     label_column= self.label_column, transform=self.valid_tfms)

            return DataLoader(
                predict_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle,
                pin_memory=True,)

    def val_dataloader(self):
        val_dataset = imageDataset(dataframe=self.valid_df, path_column= self.path_column ,
                                     label_column= self.label_column, transform=self.valid_tfms)

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

    def get_tSNE(self, data= None,  n_components=180 , perplexity=80.0  , max_count = 2500 ):
        if data == 'test':
            data = self.test_df
        elif data == 'val':
            data = self.valid_df
        else:
            data = self.train_df
        
        if data.shape[0] > max_count:
            data = data.sample(max_count)
 
        plt.figure(figsize=(15,15))
        print(f'starting tSNE for {data}')
        get_ImageDataset_tSNE(data , n_components=n_components , perplexity=perplexity ,
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


