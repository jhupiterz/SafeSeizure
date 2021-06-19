import pandas as pd
import numpy as np
import os
import scipy.io
from scipy.signal import decimate

# FOLDER_PATH might change according to your folder architecture
FOLDER_PATH = os.path.join(os.getcwd(),'..','raw_data')

def get_data():
    '''Function that extracts the data from each .mat file, 
    downsample it by a factor of 100 and export it as a .csv file'''

    # Creates a dictionary containing all file names (key=name_of_directory, value=list_of_files in directory)
    dirs = ['Patient_1', 'Patient_2', 'Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5']
    file_dict = {}
    for directory in dirs:
        files = []
        for f in os.listdir(os.path.join(FOLDER_PATH,directory,directory)):
            files.append(f)
        file_dict[directory] = files  
    print(file_dict)

    # Reads each .mat file in all directories, downsample, and export as .csv file
    for key in file_dict.keys():  
        for f in file_dict[key]:
            data = scipy.io.loadmat(os.path.join(FOLDER_PATH,key,key,f))
            segment_name = list(data.keys())[-1]
            array = decimate(decimate(data[segment_name][0][0][0],10),10) 
            dataframe = pd.DataFrame(array)
            dataframe.to_csv(f'{FOLDER_PATH}/{key}_csv/{segment_name}.csv', index = False)

def label_data():
    '''Function that adds a target (0: interictal, 1:preictal) to each csv file and exports
    as a new *_labelled.csv file'''
    files_1 = []
    files_2 = []

    dirs = ['Patient_1_csv', 'Patient_2_csv']
    for f in os.listdir(os.path.join(FOLDER_PATH,dirs[0],'train_segments_unlabelled')):
        files_1.append(f)
        
    for f in os.listdir(os.path.join(FOLDER_PATH,dirs[1],'train_segments_unlabelled')):
        files_2.append(f)

    for f in files_1:
        if f.startswith('interictal'):
            target = 0
        else:
            target = 1
        data = pd.read_csv(f'{FOLDER_PATH}/{dirs[0]}/train_segments_unlabelled/{f}')
        data['target'] = target
        f_labelled = f.strip('.csv')
        data.to_csv(f'{FOLDER_PATH}/{dirs[0]}/train_segments_labelled/{f_labelled}_labelled.csv', index = False)
    
    for f in files_2:
        if f.startswith('interictal'):
            target = 0
        else:
            target = 1
        data = pd.read_csv(f'{FOLDER_PATH}/{dirs[1]}/train_segments_unlabelled/{f}')
        data['target'] = 1
        f_labelled = f.strip('.csv')
        data.to_csv(f'{FOLDER_PATH}/{dirs[1]}/train_segments_labelled/{f_labelled}_labelled.csv', index = False)
