

sample_data_dir = "./data/sample"

supported_tabular_models = ['xgboost', 'lightgbm' , 'pycaret' , 'autogluon' , 'tabnet']
supported_tabular_problems =  ['regression' , 'classification']

supported_CNN_models = ['xgboost', 'lightgbm' , 'pycaret' , 'autogluon' , 'tabnet']
supported_CNN_problems =  ['classification' , 'segmentation']
supported_CNN_outputs = ['probability' ]


tabular_configs = {
    'xgboost' :  { 'gpu' : {'n_estimators': 100,
                                     'subsample': 0.6,
                                     'colsample_bytree': 0.9,
                                     'eta': 0.007939812697028495,
                                     'reg_alpha': 46,
                                     'reg_lambda': 64,
                                     'max_depth': 12,
                                     'min_child_weight': 20,
                                     'tree_method': 'gpu_hist',
                                     'random_state': 42},
                   'cpu' : { 'n_estimators': 500,
                                     'subsample': 0.6,
                                     'colsample_bytree': 0.9,
                                     'eta': 0.00793,
                                     'reg_alpha': 32,
                                     'reg_lambda': 32,
                                     'max_depth': 10,
                                     'min_child_weight': 5,
                                     'random_state': 42} }}



CNN_configs = {
    'resnet18' :  { 'gpu' : {'n_estimators': 100,
                                     'subsample': 0.6,
                                     'colsample_bytree': 0.9,
                                     'eta': 0.007939812697028495,
                                     'reg_alpha': 46,
                                     'reg_lambda': 64,
                                     'max_depth': 12,
                                     'min_child_weight': 20,
                                     'tree_method': 'gpu_hist',
                                     'random_state': 42},
                   'cpu' : { 'n_estimators': 1000,
                                     'subsample': 0.6,
                                     'colsample_bytree': 0.9,
                                     'eta': 0.00793,
                                     'reg_alpha': 46,
                                     'reg_lambda': 64,
                                     'max_depth': 10,
                                     'min_child_weight': 5,
                                     'random_state': 42} } ,
    'resnet50' :  { 'gpu' : {'n_estimators': 100,
                                     'subsample': 0.6,
                                     'colsample_bytree': 0.9,
                                     'eta': 0.007939812697028495,
                                     'reg_alpha': 46,
                                     'reg_lambda': 64,
                                     'max_depth': 12,
                                     'min_child_weight': 20,
                                     'tree_method': 'gpu_hist',
                                     'random_state': 42},
                   'cpu' : { 'n_estimators': 1000,
                                     'subsample': 0.6,
                                     'colsample_bytree': 0.9,
                                     'eta': 0.00793,
                                     'reg_alpha': 46,
                                     'reg_lambda': 64,
                                     'max_depth': 10,
                                     'min_child_weight': 5,
                                     'random_state': 42} } ,


}
