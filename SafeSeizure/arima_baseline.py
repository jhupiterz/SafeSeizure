import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA


def get_statistics():
    '''Function that extracts the data from each .csv file, 
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


# Check its stationarity precisely using the Augmented Dick Fuller test, and especially its p-value
y = df_average[0]
print('p-value: ', adfuller(y)[1])

# Here there is no need to do it since since no seasonal time series

# Original Series, plot autocorrelation plot (plot_acf)
fig, axes = plt.subplots(1, 2, figsize=(13,10))
axes[0].plot(y); axes[0].set_title('Original Series')
plot_acf(y, ax=axes[1])

plt.show()

#In our case, no differentiation is made (series is stationnary already, d=0) and I=6 based on the graph generated

# MA order (q) can be found by looking at the autocorrelation plot (plot_acf)
plot_acf(y);

# Based on the above graph, q=6 (already present in previous acf graph, no differentiation was needed)
# AR order (p) can be found by investigating the partial autocorrelation plot plot_pacf applied to y (diff if applicable).
plot_pacf(y, c='r');

# Based on the above graph, p=3 


## Build the model with train set 66% and test set 34%
# split into train and test sets
X = df_average[0].values
train_size = int(len(X) * 0.66)
train, test = X[:train_size], X[train_size:]

# initialize the model
# p=3, d=0 ,q=6 based on the first segment preictal file [0,400] time units. Due to computation limit, parameters are now changed
# to p=0, d=1 ,q=0
arima = ARIMA(train, order=(0, 1, 0))

# fit the model
arima = arima.fit()

# evaluate the model arima
arima.summary()

# Prepare the forecasts (predict the values of the time series based on train one)
forecast = arima.forecast(steps=len(test)) 

# Determine the mean square error
test_score = mean_squared_error(test, forecast[0])
print('Test MSE: %.3f' % test_score)

# Plot the dataset preictal_segment_1 with the forecast (forecast starting from the end of the train set and having a 
# length of the test set)
df_average[0].plot()
plt.plot(np.arange(len(df_average[0]))[train_size:], forecast[0],c='red')
plt.title('Preictal segment_1 + calculated forecast with train set 66% and test set 34%')
plt.xlabel('Time unit')
plt.ylabel('Brain activity')
plt.show()


## Build a second model with train set 90% and test set 10%
# split into train and test sets
X = df_average[0].values
train_size_2 = int(len(X) * 0.90)
train_2, test_2 = X[:train_size_2], X[train_size_2:]

# initialize the second model
# p=3, d=0 ,q=6 based on the first segment preictal file [0,400] time units. Due to computation limit, parameters are now changed
# to p=0, d=1 ,q=0
arima = ARIMA(train_2, order=(0, 1, 0))

# fit the second model
arima = arima.fit()

# evaluate the second model arima
arima.summary()

# Prepare the forecasts (predict the values of the time series based on train one)
forecast_2 = arima.forecast(steps=len(test_2)) 

# Determine the mean square error
test_score_2 = mean_squared_error(test_2, forecast_2[0])
print('Test MSE: %.3f' % test_score_2)

# Plot the dataset preictal_segment_1 with the second forecast (forecast starting from the end of the train set and having a 
# length of the test set)

df_average[0].plot()
plt.plot(np.arange(len(df_average[0]))[train_size_2:], forecast_2[0],c='red')
plt.title('Preictal segment_1 + calculated forecast with train set 90% and test set 10%')
plt.xlabel('Time unit')
plt.ylabel('Brain activity')
plt.show()
