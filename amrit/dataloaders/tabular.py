import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from amrit.utils.dataModifier import analyze
from ..models.tabular import tabularmodel


class tabularData():

    def __init__(self , data_path = None , data = None , filetype = 'csv' , target_column = None , ignore_col : list = None ):

        '''
        loads data from data_path or directly, and creates a pandas dataframe in self.data

        :param data_path:
        :param data: pandas dataframe of the data
        :param filetype:

        Pandas dtype 	Python type 	NumPy type 	Usage
        object 	str or mixed 	string_, unicode_, mixed types 	Text or mixed numeric and non-numeric values
        int64 	int 	int_, int8, int16, int32, int64, uint8, uint16, uint32, uint64 	Integer numbers
        float64 	float 	float_, float16, float32, float64 	Floating point numbers
        bool 	bool 	bool_ 	True/False values
        datetime  64 	datetime 	datetime64[ns] 	Date and time values
        timedelta[ns] 	NA 	NA 	Differences between two datetimes
        category 	NA 	NA 	Finite list of text values


        '''

        self.ignore_col = ignore_col
        self.target_column = target_column
        self.data_path = data_path

        #assert  (data!= None and  data_path != None)  , 'Provide either the data or data path'

        if data_path == None:
           self.data = data
        else:
            assert  data_path != None, "Since no data is provided, provide the path to datafile"
            if filetype == 'csv':
                self.filetype = filetype
                self.data = pd.read_csv(self.data_path)

        if self.ignore_col != None:
            self.data = self.data.drop(columns = self.ignore_col)

        self.numerical_col = self.data.select_dtypes(["float64", 'int64']).columns.tolist()
        self.datetime_col = self.data.select_dtypes(["datetime64", 'timedelta']).columns.tolist()
        self.categorical_col = self.data.select_dtypes(["bool", 'category']).columns.tolist()

    def showSample(self, length = 20):
        print(self.data.head(20))


    def analyze(self, force=False, percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], include=None, exclude=None):
        analysis = analyze(self.data , force=force, percentiles=percentiles, include=include, exclude=exclude)
        return (analysis)

    def preprocess(self, mode = 'MinMaxScaler', exclude_col : list = None , show_before_after = True):

        self.scaling_method = mode

        if mode == 'StandardScaler':
            scaler = StandardScaler()

        if mode == 'MinMaxScaler':
            scaler = MinMaxScaler()

        scaleable_columns = self.data.select_dtypes(["float64" , 'int64']).columns.tolist()

        if show_before_after == True:
            print(self.data[scaleable_columns].head(10))

        if exclude_col == None:
            data = self.data.copy()
            self.data[scaleable_columns] = scaler.fit_transform(data[scaleable_columns])
        else:
            data = self.data.copy()
            data[scaleable_columns]
            data_to_tranform = data.drop(columns = exclude_col)
            transformed_data = scaler.fit_transform(data_to_tranform)

            columns_to_include = list(set(data.columns.tolist()).symmetric_difference(set(exclude_col)))
            data[columns_to_include] = transformed_data

            self.data = data

        if show_before_after == True:
            print(f"Post scaling using - {mode}")
            print(self.data[scaleable_columns].head(10))


        self.fitted_scaler = scaler


    def  train_test_split(self, mode  = 'random' , test_size:float = 0.1 , split_random_seed:int  = 42):
        """
        modes supported
        1. 'random' - split into test train randomly as per test_size
        2. last_half - useful for chronological data - picks last (test_size*100) percent of data
        """

        if mode == 'random':
            self.test_size = test_size
            self.split_random_seed = split_random_seed
            self.train , self.test = train_test_split(self.data , test_size= test_size , random_state=split_random_seed )
            print(f"data randomly split into - {self.train.shape , self.test.shape} ")

        elif mode == 'last':
            self.test_size = test_size
            self.split_random_seed = None
            data_size = self.data.shape[0]

            split_index = data_size - int(data_size * self.test_size)
            self.train =  self.data[ :split_index]
            self.test = self.data[split_index:]
            print(f"data chronologically split into - {self.train.shape, self.test.shape} ")

        target_column = self.target_column

        self.train = self.train.reset_index()
        self.test = self.test.reset_index()

        print(self.train )

        if target_column != None:
            self.Xtrain = self.train.drop(columns=target_column)
            self.Xtest = self.test.drop(columns=target_column)
            self.Ytrain = self.train[target_column]
            self.Ytest = self.test[target_column]

            #print(self.Xtrain)

            return (self.Xtrain , self.Xtest , self.Ytrain , self.Ytest)
        else:
            return (self.train, self.test)



    def plot_continous_variables(self, subsample = True , subsample_size = 0.5 , plot_per_row = 5 ,
                             figsize=(12, 6)):

        if subsample == True:
            data = self.data.sample(int(subsample_size  * self.data.shape[0]) )
        else:
            data = self.data

        row_count = int(round(len(self.numerical_col) / plot_per_row , 0)) + 1
        fig = plt.figure(figsize= figsize )

        ax = 1
        for plotable_col in self.numerical_col :
            print(plotable_col)
            plt.subplot(row_count, plot_per_row , ax)
            sns.kdeplot(data=data, x= plotable_col ,
                        fill=True)
            # plt.set_xticks([])
            # plt.set_yticks([])
            # plt.set_xlabel('')
            # plt.set_ylabel('')
            # plt.spines['left'].set_visible(False)
            ax += 1

        fig.supxlabel('Distribution plots of diff columns', ha='center', fontweight='bold')
        fig.tight_layout()
        plt.show()

    def plot_categorical_variables(self, subsample=True, subsample_size=0.5, plot_per_row=5,
                                 figsize=(12, 6)):
        if subsample == True:
            data = self.data.sample(int(subsample_size  * self.data.shape[0]) )
        else:
            data = self.data

        row_count = int(round(len(self.categorical_col) / plot_per_row , 0)) + 1
        fig = plt.figure(figsize= figsize )

        ax = 1
        for plotable_col in self.categorical_col :
            print(plotable_col)
            plt.subplot(row_count, plot_per_row , ax)
            sns.kdeplot(data=data, x= plotable_col ,
                        fill=True)
            ax += 1

        fig.supxlabel('Distribution plots of diff columns', ha='center', fontweight='bold')
        fig.tight_layout()
        plt.show()

    def getDistributionTable(self):
        print()




    def suggestModelType(self , model_to_include = None ,  data_size = 0.2):

        """
        Trains different models on subsample of data and provides statistics for potentially best model type to use.
        """

        print(f"{data_size} data_size is being used.")

        if model_to_include == None:
            model_to_include = ['xgboost' , 'catboost' , 'lighgbm' , 'tabnet']

        print(f"model being tested are {model_to_include}")

        for model_name in model_to_include:
            if model_name == 'xgboost' or 'catboost' or 'lightgbm':
                model = tabularmodel(type = model_name)


    def add_dateTime_features(self):

        print()

    def give_me_first_look(self):
        # Xgboost based feature importance

        #Distribution visualisation
        self.showSample()
        self.analyze()
        self.plot_continous_variables()
        #self.plot_categorical_variables()

        #Dataframe sample

    def get_tSNE(self):
        from sklearn.manifold import TSNE
        import plotly.graph_objs as go

        tsne = TSNE(
            n_components=3,
            init='random',  # pca
            random_state=101,
            method='barnes_hut',
            n_iter=250,
            verbose=2,
            angle=0.5).fit_transform(np.array(self.data))

        #py.init_notebook_mode(connected=True)

        import plotly.io as pio

        pio.renderers.default = "png"

        '''pio.renderers

        Default
        renderer: 'vscode'
        Available
        renderers:
        ['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
         'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
         'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
         'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
         'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']'''

        trace1 = go.Scatter3d(
            x=tsne[:, 0],
            y=tsne[:, 1],
            z=tsne[:, 2],
            mode='markers',
            marker=dict(
                sizemode='diameter',
                # color = dataloader.data['target_carbon_monoxide'].values,
                color=self.data.index,  # z.values,,
                colorscale='Portland',
                colorbar=dict(title='duplicate'),
                line=dict(color='rgb(255, 255, 255)'),
                opacity=0.50
            )
        )

        import plotly.io as pio
        png_renderer = pio.renderers["png"]
        png_renderer.width = 500
        png_renderer.height = 500

        pio.renderers.default = "png"

        import plotly.graph_objects as go
        fig = go.Figure(
            data=[go.Bar(y=[2, 1, 3])],
            layout_title_text="A Figure Displayed with the 'png' Renderer"
        )
        fig.show()
        '''

        fig = go.Figure(
            data=[trace1],
            layout_title_text="A Figure Displayed with the 'png' Renderer"
        )
        fig.show(renderer="png")'''
        '''data = 
        layout = dict(height=800, width=800, title='test')
        fig = dict(data=data, layout=layout)
        #fig.show(renderer="svg")

        py.iplot(fig, filename='3DBubble')'''







