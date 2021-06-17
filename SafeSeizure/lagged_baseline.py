import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

def get_statistics():
    '''Function that extracts the data from each .csv file in Patient_1_csv folder, 
    already downsampled by a factor of 100'''
   
  
    # Reads each .csv file in patient_1 directory, calculate the mean for each file and concatenate each mean
    # into a new database

    temp = []
    for index in tqdm(range(1, 19)): #18 files here
        temp_df = pd.read_csv(os.path.join(os.getcwd(),f'data/Patient_1_csv/preictal_segment_{index}.csv'))
        mean_values = temp_df.mean(axis=0)
        temp.append(mean_values)
    
    return pd.concat(temp, axis=1)

df_average = get_statistics()


# Create lagged dataset between t and t+1 on preictal segment_4 file
values = df_average[[3]]
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
print(dataframe.head(5))

# split into train and test sets
X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

# persistence model
def model_persistence(x):
    return x

# walk forward validation / determine the mean square error
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
