
from src.dataloaders.tabular import tabularData
from src.dataloaders.image import imageDataset, imageDataLoader
from src.models.tabular import tabularmodel
from src.models.cnnModel import ImageModel
from src.predictions.analyze_preds import prediction_plotter
from src.utils.dataModifier import set_seed , get_labels_df_from_folder
from src.utils.augmentations import get_augmentations
from src.utils.dataModifier import trainTestSplit


from pytorch_lightning import Trainer
import warnings
warnings.filterwarnings("ignore")

train_path = "data/cat_dog1/Train"
train_df = get_labels_df_from_folder(train_path , extension='.jpg')

# train_df , val_df = trainTestSplit(train_df, target_column = None)
loader = imageDataLoader(train_df, train_df, aug_p = 0.5,
                         img_sz = 224,
                         batch_size= 64,
                         num_workers = 2, )

model = ImageModel(num_classes = 2)

#model.get_children()

model.get_params()
model.freeze(till_child = 8)
model.get_params()


module1 = imageDataLoader(train_df, train_df)
trainer = Trainer(max_epochs=1 , gpus=1)

module1.prepare_data()
trainer.fit(model, module1)