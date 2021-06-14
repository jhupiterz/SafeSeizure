import pandas as pd
import numpy as np
import os
import scipy.io
from scipy.signal import decimate

FOLDER_PATH = os.path.join(os.getcwd(),'..','raw_data')

def get_data():
    '''Function that extracts the data from each .mat file, 
    downsample it by a factor of 100 and export it as a .csv file'''
    
    # Creates list of file names
    dirs = ['Patient_1', 'Patient_2'] 
    files_1 = []
    files_2 = []

    for f in os.listdir(os.path.join(FOLDER_PATH,dirs[0],dirs[0])):
        files_1.append(f)   
    for f in os.listdir(os.path.join(FOLDER_PATH,dirs[1],dirs[1])):
        files_2.append(f)
    
    # Reads each .mat file in both directories, downsample, and export as .csv file
    for dir in dirs:
        if dir == 'Patient_1':
            files = files_1
        else:
            files = files_2    
        for f in files:
            data = scipy.io.loadmat(f'{dir}/{dir}/{f}')
            segment_name = list(data.keys())[-1]
            array = decimate(decimate(data[segment_name][0][0][0],10),10) 
            dataframe = pd.DataFrame(array)
            dataframe.to_csv(f'{dir}_csv/{segment_name}.csv', index = False)
