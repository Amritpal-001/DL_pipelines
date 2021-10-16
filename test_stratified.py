

from src.dataloaders.tabular import tabularData
from src.predictions.helper_fun import *



from src.dataloaders.tabular import tabularData
from src.predictions.helper_fun import *


data_path = './data/sample/titanic_train.csv'  #'./data/sample/TPS_july2021.csv'
target_column = 'Survived'  #'target_carbon_monoxide'
ignore_col = ['Name', 'Ticket',  'Cabin', 'Embarked'] # ['date_time' , 'target_benzene' , 'target_nitrogen_oxides']

dataloader = tabularData(data_path, target_column=target_column, ignore_col=ignore_col)

df = dataloader.data
#print(df)
df = df.dropna()
#print(filter_dfSeries_subset( df['PassengerId'] , df['Sex'], search_tag = 'male'))


print(compute_stratified_metrics(df['Age'] , df['PassengerId'] , df['Sex'] ))