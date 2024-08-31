import pandas as pd
import numpy as np

def normalize_data(X, method=None, random_offset_value=35):

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    numeric = X.select_dtypes(include=['int64', 'float64'])
    non_numeric = X.select_dtypes(include=['category', 'object', 'bool'])

    if method is None:
        return X

    if method == "demean":
        # Center around mean
        mean = numeric.mean(axis=0)
        normalized_numeric = numeric - mean
    
    elif method == "median":
        # Center around median
        median = numeric.median(axis=0)
        normalized_numeric = numeric - median

    elif method == "min_max":
        # Min-Max scaling to range [0, 1]
        min_val = numeric.min(axis=0)
        max_val = numeric.max(axis=0)
        normalized_numeric = (numeric - min_val) / (max_val - min_val)
    
    elif method == "random_offset":
        # Subtract a constant offset
        normalized_numeric = numeric - random_offset_value
    
    else:
        raise ValueError(f"Normalization method '{method}' is not recognized.")
    

    normalized_numeric = normalized_numeric.reset_index(drop=True)
    non_numeric = non_numeric.reset_index(drop=True)
    normalized_X = pd.concat([normalized_numeric, non_numeric], axis=1)
    normalized_X = pd.DataFrame(normalized_X)

    return normalized_X
