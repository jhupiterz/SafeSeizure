# Main function that iterates ICA + wavelet transform on all Dogs directories

import wavelet_utils
import pywt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import FastICA
import os

FOLDER_PATH = os.path.join('../raw_data/')
DIRS = ['Dog_1_csv', 'Dog_2_csv', 'Dog_3_csv', 'Dog_4_csv', 'Dog_5_csv']

ICA_TRANSFORMER = FastICA(n_components= 10, max_iter= 3000000, random_state=0, tol=0.1)
WAVELET = pywt.Wavelet('db4')

def get_X_y(path = FOLDER_PATH, ica = ICA_TRANSFORMER, wavelet = WAVELET):
    # For each Dog_csv file: (1) create list of files -> (2) reads each file
    #                        (3) ICA Transform -> (4) DWT -> (5) Extracts features
    #                        (4) Creates target column
    for directory in DIRS[0:2]:
        folder_path = os.path.join(path,directory)
        files = []
        data = []
        y = []
        for f in os.listdir(folder_path):
            if f.startswith('preictal') | f.startswith('interictal'):
                files.append(f)
        features = []
        target = []
        for file in files:
            segment = pd.read_csv(os.path.join(folder_path, file))
            if (directory == 'Dog_1_csv') | (directory == 'Dog_2_csv') | (directory == 'Dog_3_csv') | (directory == 'Dog_4_csv'):
                segment.drop([15], axis = 0, inplace = True)
            segment_transformed = ica.fit_transform(segment)
            coeffs = pywt.wavedec(segment_transformed, wavelet, level = 7)
            segment_features = wavelet_utils.feature_extraction(coeffs)
            np.asarray(features.append(segment_features))
            if file.startswith('interictal'):
                target.append(0)
            elif file.startswith('preictal'):
                target.append(1)
            if directory == 'Dog_1_csv':
                features_dog1 = features
                target_dog1 = target
            if directory == 'Dog_2_csv':
                features_dog2 = features
                target_dog2 = target
            if directory == 'Dog_3_csv':
                features_dog3 = features
                target_dog3 = target
            if directory == 'Dog_4_csv':
                features_dog4 = features
                target_dog4 = target
            if directory == 'Dog_5_csv':
                features_dog5 = features
                target_dog5 = target
            #elif directory == 'Patient_1_csv':
                #features_pat = features
                #target_pat = target
        target = pd.Series(np.array(target))
    data = np.vstack((features_dog1, features_dog2))#, features_dog3, features_dog4, features_dog5))
    y = pd.concat([pd.Series(target_dog1), pd.Series(target_dog2)], axis = 0, ignore_index = True)#, pd.Series(target_dog3), 
                #pd.Series(target_dog4), pd.Series(target_dog5)], 
                #axis = 0, ignore_index = True)
    X = data.reshape(len(data),-1)
    print(len(X), len(y))
    return X, y

def get_scaled_split_data():
    robust = RobustScaler()
    X, y = get_X_y()
    print('Retrieved X, y successfully')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_scaled = robust.fit_transform(X_train)
    X_test_scaled = robust.transform(X_test)
    return X_train_scaled, y_train, X_test_scaled, y_test
