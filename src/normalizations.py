import pandas as pd
import numpy as np

def mean_normalize(x):
    return (x - np.mean(x, axis=0))

def median_normalize(x):
    return (x - np.median(x, axis=0))

def min_max_normalize(x):
    return (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))