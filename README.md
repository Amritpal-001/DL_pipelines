
# KagglePipeLine


## Model Supported
### tabular models
- type = "xgboost"
- problem = "classification" , "regression"
- output = 'probabilities' , 'classes'

### CNN models
- type =  all timm models
- problem = "classification"
- output = 'probabilities' , 'classes'


## Directory
```bash
├── amrit
│ ├── configs
│ ├── constants.py
│ ├── credentials
│ ├── dataloaders
│ │ ├── dimension_Reduction.py
│ │ ├── image.py
│ │ └── tabular.py
│ ├── explain
│ │ ├── explain_cnn.py
│ │ ├── explain_helpers.py
│ │ ├── feature_maps.py
│ │ └── saliency
│ ├── metrics
│ │ ├── classification_metrics.py
│ │ ├── meter.py
│ │ ├── regression_metrics.py
│ │ └── segmentation.py
│ ├── models
│ │ ├── cnnModel.py
│ │ └── tabular.py
│ ├── predictions
│ │ ├── analyze_preds.py
│ │ ├── helper_fun.py
│ ├── templates
│ └── utils
│     ├── augmentations.py
│     ├── dataModifier.py
│     ├── dicoms.py
│     ├── dimensionality.py
│     ├── DownloadData.py
│     ├── featureSelection.py
│     ├── filefolder.py
│     ├── image_registration.py
│
├── examples
│ ├── example_cnn.py
│ ├── example_hydra.py
│ └── example_tabular.py
│
├── readme.md
├── requirements.txt
├── tests
└── wishlist.md
```


