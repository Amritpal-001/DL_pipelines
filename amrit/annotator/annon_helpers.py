
import glob

import numpy as np
import pandas as pd


def get_filepath_list(folder_path : str = None, file_extension = '.dcm' ,  recursive=False):
    assert folder_path != None , "please provide a folder path" 
    
    if recursive ==True:
        file_list = glob.glob(folder_path + f'/**/*{file_extension}' , recursive=True)
    else:
        file_list = glob.glob(folder_path + f'/*{file_extension}')
    
    file_list = sorted(file_list)
    print('Number of files in folder -  ', len(file_list))
    return(file_list)


def get_filepath_df(folder_path : str = None, file_extension = '.dcm' ,  recursive=False):

    filepath_list = get_filepath_list(folder_path  = folder_path, file_extension = file_extension,
                      recursive=recursive)

    df = pd.DataFrame(filepath_list, columns = ['filepath'])

    return(df)


def generate_psuedo_labels(df, label_col_name = 'label', label_list =[0,1] , p = None ):

    if p != None:
        p = p
    
    df[label_col_name]  = np.random.choice(label_list, size=df.shape[0], p=p)

    return(df)
