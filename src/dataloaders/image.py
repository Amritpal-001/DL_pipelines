from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import LightningDataModule
import numpy as np
import matplotlib.pyplot as plt

import torch
from src.utils.augmentations import get_augmentations

classes_dict = {
    'Dogs': 0,
    'Cats': 1,
    'Test11':2,
    'Test222': 3,
}
def target_transformations(x):
    return torch.tensor(classes_dict.get(x))


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

        self.classes = self.labels.unique

        assert len(self.labels) == len(self.paths), f"{path_column} and {label_column} should have same length"
        print("data size - ", len(self.paths))
        print('classes = ', self.classes)


    def __getitem__(self, index):
        file_path = self.paths.iloc[index]
        label = self.labels.iloc[index]
        image = Image.open(file_path)
        if self.transform is not None:
            image = self.transform(image)
            label = target_transformations(label)
        return image, label

    def __len__(self):
        return len(self.paths)


class imageDataLoader(LightningDataModule):
    def __init__(
            self,
            train_df=None,
            valid_df=None,
            aug_p: float = 0.5,
            img_sz: int = 124,
            batch_size: int = 16,
            num_workers: int = 4
    ):
        super().__init__()

        self.train_df, self.valid_df = train_df, valid_df
        self.aug_p = aug_p
        self.img_sz = img_sz
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_tfms, self.valid_tfms = get_augmentations(self.aug_p, image_size=self.img_sz)

    def prepare_data(self):
        self.train_df = self.train_df  # imageDataset(train_df , transform = self.train_tfms)
        self.valid_df = self.valid_df  # imageDataset(valid_df, transform = self.valid_tfms)

    def train_dataloader(self):
        train_dataset = imageDataset(dataframe=self.train_df, transform=self.train_tfms)

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        val_dataset = imageDataset(dataframe=self.valid_df, transform=self.valid_tfms)

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

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
        #
        # imshow(out)

