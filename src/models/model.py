


from ..constants import supported_CNN_models , supported_CNN_outputs, supported_CNN_problems
from ..constants import tabular_configs , supported_tabular_models  , supported_tabular_problems


class Model():
    def __init__(self , architecture , problem  , output , device ):
        self.problem = None #type of problem - regression/classification
        self.architecture = None  #model architecture e.g. - Xgboost, Resnet,
        self.device = None
        self.output = None

        self.config = None


    def check_inputs(self):

        if self.problem == 'regression' or 'classification':
            assert self.problem in supported_tabular_problems, f"Supports only {supported_tabular_problems} tasks"
            assert self.architecture in supported_tabular_models, f"Supports only {supported_tabular_models}"

            print("inputs are good")

        elif self.problem == 'imageClassification':
            assert self.problem in supported_CNN_problems, f"Supports only {supported_CNN_problems} tasks"
            assert self.architecture in supported_CNN_models, f"Supports only {supported_CNN_models}"

            print("inputs are good")

