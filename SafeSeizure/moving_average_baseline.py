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


# calculate the moving average on preictal segment_4 file and add the column SMA_150 (window = 150)
df_average['SMA_150'] = df_average.iloc[:,3].rolling(window=150).mean()


# plot the datasets moving average and the file preictal_segment_4 from 0 to 400 time units. Moving average has been set with a window of 150 time units.
plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(df_average.iloc[:400, 0],label='data')
plt.plot(df_average['SMA_150'][150:400],label='SMA 150 time units')
plt.legend(loc=2)

# split into train and test sets
X = df_average[[3,'SMA_150']].values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

# Determine the mean square error
test_score = mean_squared_error(test_y, test_X)
print('Test MSE: %.3f' % test_score)