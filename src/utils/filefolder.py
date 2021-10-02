
import glob
import os.path
import pandas as pd

def get_file_list(folder_path : str = None ,extension = '.dcm', recursive = False):
    assert folder_path != None , "please provide a folder path"
    if recursive == False:
        file_list = glob.glob(folder_path + './*' +extension , recursive=True)
    else:
        file_list = glob.glob(folder_path + '/**/*' +  extension , recursive=True)
    file_list = sorted(file_list)
    
    return(file_list)


def get_folder_dataframe(folder_path: str = None, extension='.dcm', recursive=False):
    file_list = get_file_list(folder_path = folder_path ,extension = extension , recursive = recursive)

    df = pd.DataFrame(columns=['file_name', 'metadata'])

    for x in range(len(file_list)):
        file_name = file_list[x]
        df = df.append(pd.Series([os.path.basename(file_name),  os.path.dirname(file_name) ], index=df.columns), ignore_index=True)
    return (df)
