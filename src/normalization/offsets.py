import pandas as pd
import numpy as np

def normalize_data(X, method=None, random_offset_value=35):
    """Normalize data using different methods."""
    if method == None:
        return X
    elif method == "demean":
        print("mean of the data: ", X.mean(axis=0))
        return X - X.mean(axis=0)
    elif method == "median":
        print("median of the data: ", X.median(axis=0))
        return X - X.median(axis=0)
    elif method == "random":
        return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    elif method == "random_offset":
        return (X - random_offset_value)
    else:
        raise ValueError(f"Normalization method '{method}' is not recognized.")
