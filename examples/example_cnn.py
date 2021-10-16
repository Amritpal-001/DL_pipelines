import torch
import random
import warnings
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import sys
sys.path.insert(0 , '../Kaggle Pipeline/')
print(sys.path)

from amrit.explain.explain_helpers import load_image
from amrit.dataloaders.image import imageDataset, imageDataLoader
from amrit.models.cnnModel import ImageModel
from amrit.predictions.analyze_preds import prediction_plotter
from amrit.utils.dataModifier import set_seed , get_labels_df_from_folder, trainTestSplit
from amrit.utils.dataModifier import trainTestSplit
from amrit.utils.filefolder import get_random_Imagepath , get_random_list_Imagepath
from amrit.explain.explain_cnn import get_guided_GradCAM , plot_gradcam_comparison, plot_in_grid , plot_gradcam_single_image

warnings.filterwarnings("ignore")

# Create df
train_path = "../data/cat_dog/Train"
train_path = "../data/medical"
train_df = get_labels_df_from_folder(train_path , extension='.jpg')
train_df , val_df = trainTestSplit(df=train_df, mode='random', test_size = 0.5, split_random_seed= 42)
test_df , val_df = trainTestSplit(df=val_df, mode='random', test_size = 0.5, split_random_seed= 42)

# Datalaoder from df
loader = imageDataLoader(train_df, val_df, test_df = test_df , aug_p = 0.5,
                         img_sz = 32, batch_size= 32, num_workers = 2, )

# tSNE data
loader.get_tSNE('test')

loader.compare_TSNE()

## Model
model = ImageModel(num_classes = 5, pretrained= True)
#model.get_children()
model.freeze(till_child = 8)
model.get_params()

# Pytorch lightning Trainer
checkpoint_callback = ModelCheckpoint(monitor="val_loss" , save_last=True, save_top_k=2)
trainer = Trainer(max_epochs=1 , gpus=1, callbacks=[checkpoint_callback])
trainer.fit(model, loader )

### Make predictions
#test = loader.test_dataloader()
#preds = make_predictions(dataloader = test , model = model)
predictions = trainer.predict(model, loader)
print(predictions)
# output = trainer.test(model , loader)
# print(output)


# Single Image GradCAM
'''
image = load_image(image_path, size=224)
get_guided_GradCAM(image , model , 1) 
plot_gradcam_comparison(image_path , model , image_class_id= 0 )
plot_gradcam_single_image(image_path, model, image_size = 224 ,  image_class_id=0)
'''

# Batch GradCAM for Randomaly selected scans
image_list , label_list = get_random_list_Imagepath(train_df, 25)
i = 2
plot_in_grid(image_list,  plot_gradcam_single_image, model , label_list , grid_size = 5 , cut_with_mask = False , image_class_id= i , percentile=70 )
plot_in_grid(image_list,  plot_gradcam_single_image, model , label_list , grid_size = 5 , cut_with_mask = True , image_class_id= i , percentile=70 )

'''
for i in range(0,5):
     plot_in_grid(image_list,  plot_gradcam_single_image, model , label_list , grid_size = 5 , cut_with_mask = False , image_class_id= i , percentile=60 )
     plot_in_grid(image_list,  plot_gradcam_single_image, model , label_list , grid_size = 5 , cut_with_mask = True , image_class_id= i , percentile=60 )
'''

## Feature maps visualisation
test_dataloader = loader.test_dataloader()
input = next(iter(test_dataloader))[0]
#image = torch.randn(10,3,64,64)

model.unfreeze()
model.visualize_feature_maps(input , show_forward = True , class_index = 0,
                               show_backward =True,  layer = 'conv1' , layer_sub_index = 0)
