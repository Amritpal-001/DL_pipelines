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

# Create df
train_path = "data/cat_dog1/Train"
train_path = "data/medical"
train_df = get_labels_df_from_folder(train_path , extension='.jpg')
train_df , val_df = trainTestSplit(df=train_df, mode='random', test_size = 0.5, split_random_seed= 42)
test_df , val_df = trainTestSplit(df=val_df, mode='random', test_size = 0.5, split_random_seed= 42)

# Datalaoder from df
loader = imageDataLoader(train_df, val_df, test_df = test_df , aug_p = 0.5,
                         img_sz = 128, batch_size= 32, num_workers = 2, )

## Model
model = ImageModel(num_classes = 5, pretrained= True)
#model.get_children()
model.freeze(till_child = 8)
model.unfreeze()
model.get_params()

# Pytorch lightning Trainer
checkpoint_callback = ModelCheckpoint(monitor="val_loss" , save_last=True, save_top_k=2)
trainer = Trainer(max_epochs=1 , gpus=1, callbacks=[checkpoint_callback])
#trainer.fit(model, loader )

from torchvision import transforms

print(dict(model.timm_model.named_children()).keys())
image_list , label_list = get_random_list_Imagepath(train_df, 24)

image = Image.open(image_list[0])
trans = transforms.Compose([transforms.ToTensor()])
image = trans(image)

test_dataloader = loader.test_dataloader()
input = next(iter(test_dataloader))[0]
#image = torch.randn(10,3,64,64)

model.unfreeze()
model.visualize_feature_maps(input , show_forward = False , class_index = 0 *-,
                               show_backward =True,  layers = ['layer1' , 'conv1'] , layer_sub_index = 0)
