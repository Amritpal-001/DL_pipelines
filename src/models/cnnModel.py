
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

import matplotlib.pyplot as plt

from src.explain.feature_maps import create_forward_hook , create_backward_hook , visualize_feature_maps

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
        loss_fn: str = "cross_entropy",
        lr=1e-4,
        wd=1e-6,
        pretrained = True,
    ):
        super().__init__()

        self.timm_model = timm.create_model(model_name=model_name, pretrained=pretrained
                                          ,num_classes=num_classes , in_chans=3)

        self.loss_fn = loss_fns[loss_fn]
        self.lr = lr
        self.accuracy = pl.metrics.Accuracy()
        self.wd = wd
        self.num_classes = num_classes

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
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat, y), prog_bar=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):

        return optim.AdamW(self.parameters(), lr=self.lr
                           , weight_decay=self.wd )

    def print_timmmodels(self , name):
        print(timm.list_models(f'*{name}*'))

    def get_children_name(self):
        return dict(self.timm_model.named_children()).keys()

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

    def visualize_feature_maps(self , image , show_forward = True , class_index = 1,
                               show_backward =False,  layers = ['layer1'] , layer_sub_index = 0):

        index = layer_sub_index

        if show_forward == True:
            forward_hook = create_forward_hook(self.timm_model)
        if show_backward == True:
            backward_hook = create_backward_hook(self.timm_model)

        output = self.timm_model(image)
        print(output)
        if show_forward == True:
            for layer_name in layers:
                visualize_feature_maps(forward_hook[layer_name][index][index])

        if show_backward == True:
            output.sigmoid()
            self.timm_model.zero_grad()
            class_tensor = [0] * self.num_classes
            class_tensor[class_index] = 1
            print(class_tensor)
            one_hot = torch.tensor(class_tensor).float().requires_grad_(True)
            one_hot.mul(output).sum().backward(retain_graph=True)

            for layer_name in layers:
                visualize_feature_maps(backward_hook[layer_name][0][0][0])




document_dic = {
    'ModelCheckpoint' : 'ModelCheckpoint(dirpath=None, filename=None, monitor=None, verbose=False,'
                        'save_last=None, save_top_k=1, save_weights_only=False,'
                        'mode="min", auto_insert_metric_name=True,'
                        'every_n_train_steps=None, train_time_interval=None,'
                         'every_n_epochs=None, save_on_train_epoch_end=None,'
                        'period=None, every_n_val_epochs=None)'
}
def print_docs(function):
    print(document_dic[function])



def make_model_predictions( test_dataloader , model_path = None , model = None, device = 'cuda', return_probs = True):

    if model_path != None:
        model_load(model_path)

    was_training = model.training
    model.eval()
    test_list = []
    model.to(device)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            if return_probs == False:
                _, preds = torch.max(outputs, 1)
            test_list.append(outputs)
            print(outputs.shape)

        model.train(mode=was_training)
    return(test_list)
