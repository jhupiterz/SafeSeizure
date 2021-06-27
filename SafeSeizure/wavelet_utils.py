# Functions used in the ICA + wavelet transform wavelet.py
import numpy as np
from pyentrp import entropy
from scipy.stats import median_absolute_deviation
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, make_scorer
from scipy.signal import spectrogram
from scipy.signal import resample
import warnings
warnings.filterwarnings('ignore')

def flatten_list(l):
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def mean_absolute_value(array):
    absolute_values = []
    for i in range(len(array)):
        absolute_values.append(abs(array[i]))
    mean_absolute_value = np.asarray(absolute_values).sum()*(1/len(array))
    #print(mean_absolute_value)
    return mean_absolute_value

def average_power(array):
    average_power_values = []
    for i in range(len(array)):
        average_power_values.append(abs(array[i])**2)
    average_power_value = np.asarray(average_power_values).sum()*(1/len(array))
    #print(average_power_value)
    return average_power_value

def shan(d1):
    sh1=[]
    d1=np.rint(d1)
    for i in range(d1.shape[0]):
        X=d1[i]
        sh1.append(entropy.shannon_entropy(X))
    return(sh1)

def feature_extraction(decomposed_signals):
    mav = []
    avp = []
    std = []
    var = []
    mean = []
    for coeff in decomposed_signals:
        mav_electrode = []
        avp_electrode = []
        std_electrode = []
        var_electrode = []
        mean_electrode = []
        for electrode in coeff:
            mav_electrode.append(mean_absolute_value(electrode))
            avp_electrode.append(average_power(electrode))
            std_electrode.append(np.std(electrode))
            var_electrode.append(np.var(electrode))
            mean_electrode.append(np.mean(electrode))
        mav.append(mav_electrode)
        avp.append(avp_electrode)
        std.append(std_electrode)
        var.append(var_electrode)
        mean.append(mean_electrode)
    mav = flatten_list(mav)
    avp = flatten_list(avp)
    std = flatten_list(std)
    var = flatten_list(var)
    mean = flatten_list(mean)
    shan_ent = []
    for i in range(len(decomposed_signals)):
        shan_ent.append(shan(decomposed_signals[i]))
    shan_ent = flatten_list(shan_ent)
    data = np.vstack((np.array(mav), np.array(avp), 
                     np.array(std), np.array(var), np.array(mean), np.array(shan_ent)))
    #data = pd.DataFrame(data)
    features = data.T
    #features.columns = ['mean_abs_value', 'average_power', 'std', 'var', 'mean', 'shannon_entropy']

    return features
