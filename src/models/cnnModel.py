
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from torch.optim import optimizer
from pathlib import Path
import pandas as pd
#from data import *
import torch.nn.functional as F

from torch.nn import BCELoss, CrossEntropyLoss
import torch
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from torch import optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# import hydra
# from hydra.utils import instantiate
# from omegaconf import DictConfig, OmegaConf
import timm

loss_fns = {"binary_cross_entropy_with_logits": F.binary_cross_entropy_with_logits,
    "binary_cross_entropy" :F.binary_cross_entropy,
    'cross_entropy' :F.cross_entropy,}

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST , FashionMNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np


class ImageModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = 'resnet18',
        num_classes: int = 10,
        loss_fn: str = "binary_cross_entropy",
        lr=1e-4,
        wd=1e-6,
        pretrained = False,
    ):
        super().__init__()

        self.timm_model = timm.create_model(model_name=model_name, pretrained=pretrained
                                          ,num_classes=num_classes , in_chans=3)

        self.loss_fn = loss_fns[loss_fn]
        self.lr = lr
        self.accuracy = pl.metrics.Accuracy()
        self.wd = wd

    def forward(self, x):
        z = self.timm_model(x)
        return z

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) #.view(-1)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) #.view(-1)
        loss = self.loss_fn(y_hat, y)
        self.log("valid_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat, y), prog_bar=True)

    def configure_optimizers(self):

        return optim.AdamW(self.parameters(), lr=self.lr
                           , weight_decay=self.wd )

    def get_children(self):

        child_count = 0
        for child in self.timm_model.children():
            print("child_count - " , child_count)
            print(child)
            print()
            print()
            print()
            child_count +=1

    def freeze_internal(self, till_child = 2) :
        ct = 0
        for child in self.children():
            ct += 1
            if ct < till_child:
                for param in child.parameters():
                    param.requires_grad = False

    def freeze(self, till_child = 2) :
        ct = 0
        for child in self.timm_model.children():
            ct += 1
            if ct < till_child:
                for param in child.parameters():
                    param.requires_grad = False


    def get_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.timm_model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        if params > 100000:
            Trainable_params = ('Trainable params- ' + str(round(params / 1000000, 2)) + ' million')
            print(Trainable_params)
        else:
            Trainable_params = ('Trainable params - ' + str(params))
            print(Trainable_params)


