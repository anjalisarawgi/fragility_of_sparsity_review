import pandas as pd
import numpy as np

def normalize_data(X, method=None, random_offset_value=35):
    print("x shape inside normalize_data: ", X.shape)
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    # Select the numerical columns
    numeric = X.select_dtypes(include=['int64', 'float64'])
    print("Numerical columns: ", numeric.columns)
    print("Number of numerical columns: ", len(numeric.columns))
    
    # Select the non-numeric columns
    non_numeric = X.select_dtypes(include=['category', 'object', 'bool'])
    print("Non-numeric columns: ", non_numeric.columns)
    print("Number of non-numeric columns: ", len(non_numeric.columns))

    if method is None:
        return X

    # Normalize numerical data based on the selected method
    if method == "demean":
        # Center around mean
        mean = numeric.mean(axis=0)
        normalized_numeric = numeric - mean
        print("Mean of the data: ", mean)
    
    elif method == "median":
        # Center around median
        median = numeric.median(axis=0)
        normalized_numeric = numeric - median
        print("Median of the data: ", median)

    elif method == "min_max":
        # Min-Max scaling to range [0, 1]
        min_val = numeric.min(axis=0)
        max_val = numeric.max(axis=0)
        normalized_numeric = (numeric - min_val) / (max_val - min_val)
        print("Min values of the data: ", min_val)
        print("Max values of the data: ", max_val)
    
    elif method == "random":
        # Min-Max scaling to range [0, 1]
        min_val = numeric.min(axis=0)
        max_val = numeric.max(axis=0)
        normalized_numeric = (numeric - min_val) / (max_val - min_val)
        print("Min values of the data: ", min_val)
        print("Max values of the data: ", max_val)
    
    elif method == "random_offset":
        # Subtract a constant offset
        normalized_numeric = numeric - random_offset_value
        print(f"Subtracting a random offset value: {random_offset_value}")
    
    else:
        raise ValueError(f"Normalization method '{method}' is not recognized.")
    
    # Concatenate the normalized numerical data with the non-numeric columns
    # normalized_X = pd.concat([normalized_numeric, non_numeric.reset_index(drop=True)], axis=1)
    # print("Normalized data shape: ", normalized_X.shape)

    # Reset index to ensure alignment before concatenation
    normalized_numeric = normalized_numeric.reset_index(drop=True)
    non_numeric = non_numeric.reset_index(drop=True)
    
    # Concatenate the normalized numerical data with the non-numeric columns
    normalized_X = pd.concat([normalized_numeric, non_numeric], axis=1)
    print("Normalized data shape: ", normalized_X.shape)
    return normalized_X
