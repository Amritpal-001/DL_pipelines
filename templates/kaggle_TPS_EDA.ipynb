
from src.models.tabular import tabularmodel
from src.dataloaders.tabular import tabularData
from src.predictions.analyze_preds import prediction_plotter
from src.utils.dataModifier import set_seed

import warnings
warnings.filterwarnings("ignore")

preprocess = True
preprocessMode = 'MinMaxScaler'
train_test_split = True
test_train_split_Mode = 'random'
test_size =  0.1

data_path = './data/sample/TPS_july2021.csv'
target_column = 'target_carbon_monoxide'
ignore_col = ['date_time' , 'target_benzene' , 'target_nitrogen_oxides'] #['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

architecture = "xgboost"
problem =  "regression"
num_classes =  1
accelerator =  None

dataloader = tabularData(data_path, target_column=target_column, ignore_col=ignore_col)
dataloader.analyze()
#dataloader.plot_continous_variables(subsample=False, subsample_size=0.5, plot_per_row=5)
dataloader.preprocess(show_before_after=True)

model = tabularmodel(architecture=architecture, problem=problem)
model.fit(dataloader)
predictions = model.predict()

model.plot_train_metrics()

plotter = prediction_plotter(predictions, dataloader)
plotter.compute_metrics()

