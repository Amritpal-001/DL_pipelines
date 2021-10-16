

from amrit.predictions.helper_fun import *
import pandas as pd

from amrit.metrics.meter import get_classification_statistics

data_path = './data/sample_pred/sample_pred.csv'  #'./data/sample/TPS_july2021.csv'
data_path = './data/sample_pred/sample_multi_class_pred.csv'  #'./data/sample/TPS_july2021.csv'

df = pd.read_csv(data_path)

preds = df['preds_prob_3']
gt = df['gt']

output = get_classification_statistics(preds, gt)
#print(output)
#preds_prob = df['preds_prob']

from amrit.utils.dataModifier import analyze

#print(get_help_index_functions())
#print(get_TP_index(preds , gt , true_class= 3, ))
'''print(get_FP_index(preds , gt))
print(get_FN_index(preds , gt))
print(get_TN_index(preds , gt))
print(analyze(preds_prob))
print(get_FarNorth_index(preds_prob , gt))
print(get_FarSouth_index(preds_prob , gt))'''