import numpy as np


def minmax_scale(X):
    data_min = X.min(axis=0)
    data_max = X.max(axis=0)
    
    scale = data_max - data_min
    scale[scale == 0] = 1
    
    return (X - data_min) / scale