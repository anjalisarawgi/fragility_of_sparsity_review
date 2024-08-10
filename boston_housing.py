import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import MinMaxScaler
from src.normalizations import mean_normalize, median_normalize, min_max_normalize
from src.models.ols import ols
from src.models.lasso import lasso

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

feature_names = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
    ]

# ols and lasso on original data
ols(data, target, feature_names)
lasso(data, target, feature_names)



