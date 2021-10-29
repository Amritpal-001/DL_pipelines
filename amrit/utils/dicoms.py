#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os

import pandas as pd
import pydicom


# In[10]:

def get_metadata_from_dicom(dcm_path: str) :
    """
    extract the metadata from a dicom file
    Args:
        dcm_path: path to dicom

    Returns:
        dicom metadata as dicom dataset
    """
    assert os.path.exists(dcm_path)
    try:
        dcm_meta = pydicom.read_file(dcm_path, stop_before_pixels=True, force=True)
        return dcm_meta
    except Exception as e:
        logger.exception("unable to read metadata from dicom")
        raise RuntimeError("unable to read metadata from dicom") from e
        

def get_file_list(folder_path : str = None ,extension = '.dcm', recursive = False):
    assert folder_path != None , "please provide a folder path"
    if recursive == False:
        file_list = glob.glob(folder_path + './*' +extension , recursive=True)
    else:
        file_list = glob.glob(folder_path + '/**/*' +  extension , recursive=True)
    file_list = sorted(file_list)
    
    return(file_list)

def extract_dicom_metadata_tags( folder_path : str = None, tag_list :list = None,
                                csv_name : str = 'metadata.csv'):
    assert folder_path != None , "Please provide a folder path"

    file_list = get_file_list(folder_path)
    numOfFiles = len(file_list)

    metadata_list =[]

    for x in range(numOfFiles):
            file_name = file_list[x]
            metadata = get_metadata_from_dicom(file_name)
            if tag_list != None:
                            temp_list = []
                            for tag in tag_list:
                                try:
                                    temp_list.append(metadata[tag].value)
                                except:
                                    temp_list.append("nan")
                        
            metadata_list.append([os.path.basename(file_name)] + temp_list)
            if x % 2500 == 0:
                print(x)
    
    metadata_df = pd.DataFrame(metadata_list, columns = ['file_name'] + tag_list)
    
    metadata_df.to_csv(csv_name , index = False)
    return(metadata_df)


def get_all_dicom_metadata(folder_path : str = None , csv_name : str = 'metadata.csv'):
    
    df = pd.DataFrame(columns = ['file_name' , 'metadata']) 
    df

    for x in range(len(dicom_files)):
        file_name = dicom_files[x]
        metadata = get_metadata_from_dicom(file_name)
        df = df.append(pd.Series([os.path.basename(file_name) , [metadata] ] , index = df.columns), ignore_index=True )
        if x % 2499 == 0:
            print(x)

    df.to_csv(csv_name , index = None)
    
    return(df)


# In[12]:


dicom_files = get_file_list(folder_path = "./QURE_ME")


# In[14]:


len(dicom_files)


# In[45]:


def convert_to_png(file_path , destination_path = './raw_files/png/'):
    try:
        pixel = pydicom.dcmread(file_path).pixel_array
        try:    
            img = Image.fromarray(pixel)
            img.save(destination_path + os.path.basename(file_path)[:-4] + '.png')
        except:
            print(file_path , "cant generate Image from pixel" , pixel,  )
    except:
        print("no_pixel_data_found" , file_path)
        
    
def convert_folder_to_png(folder_path = "./QURE_ME" , destination_path = './raw_files/png/'):    
    dicom_files = get_file_list(folder_path)
    
    for index in range(600,10000): #len(dicom_files)):
        file_name = dicom_files[index]
        convert_to_png(file_name , destination_path = destination_path)
        
        if index % 1000 == 0:
            print(index , "images converted to PNG")
            
list_on_errors = convert_folder_to_png(folder_path = "./QURE_ME" , destination_path = './raw_files/png/')


# In[ ]:


get_ipython().system('ls raw_files/png')

