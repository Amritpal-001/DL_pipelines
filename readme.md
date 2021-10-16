
for any function documentaion - use
 - help(function_name)


amritpal@amritpal-TUF-Gaming-FX506LI-FX566LI:~/PycharmProjects/Kaggle Pipeline (copy)$ tree
```bash
├── amrit
│ ├── configs
│ │ ├── imageClassification.yaml
│ │ ├── Optuna.yaml
│ │ └── tabular.yaml
│ ├── constants.py
│ ├── credentials
│ │ ├── kaggle
│ │ └── weight_and_biases
│ ├── dataloaders
│ │ ├── dimension_Reduction.py
│ │ ├── image.py
│ │ ├── __init__.py
│ │ └── tabular.py
│ ├── explain
│ │ ├── explain_cnn.py
│ │ ├── explain_helpers.py
│ │ ├── feature_maps.py
│ │ ├── __init__.py
│ │ └── saliency/
│ ├── metrics
│ │ ├── classification_metrics.py
│ │ ├── meter.py
│ │ ├── regression_metrics.py
│ │ └── segmentation.py
│ ├── models
│ │ ├── cnnModel.py
│ │ ├── __init__.py
│ │ ├── model.py
│ │ ├── __pycache__
│ │ │ ├── __init__.cpython-38.pyc
│ │ │ ├── model.cpython-38.pyc
│ │ │ └── tabular.cpython-38.pyc
│ │ └── tabular.py
│ ├── predictions
│ │ ├── analyze_preds.py
│ │ ├── helper_fun.py
│ │ └── __init__.py
│ ├── __pycache__
│ │ ├── constants.cpython-38.pyc
│ │ └── __init__.cpython-38.pyc
│ ├── templates
│ │ ├── __init__.py
│ │ ├── kaggle_TPS_dimension_reduction.ipynb
│ │ ├── kaggle_TPS_EDA.ipynb
│ │ ├── kaggle_TPS_EDA.py
│ │ ├── kaggle_TPS_inference.py
│ │ └── kaggle_TPS_train.py
│ └──templates
├── example_cnn.py
├── example_hydra.py
├── example_tabular.py
├── readme.md
├── requirements.txt
├── test_cnn.py
├── test_dataloader.py
├── test_feature_maps.py
├── testing.py
├── test_metrics.py
├── test_module.py
├── test_stratified.py
├── test_tabular.log
├── test_tsne.py
└── Untitled.ipynb

```

## Directories
./src/
    ../dataloaders/
        tabular.py
        image.py #2D
    models/
        tabularModel.py
        cnnModel.py
    explain/
        explain_cnn - GradCAM
        feature_maps - Feature maps
    dim
    constants.py - directories, model_weights_path
    config/
    credentials/
            weights_and_biases
            kaggle

## Model types
### tabular models
- type = "xgboost" 
- problem = "classification" , "regression"
- output = 'probabilities' , 'classes'

### CNN models
- type =  all timm models
- problem = "classification" 
- output = 'probabilities' , 'classes'


