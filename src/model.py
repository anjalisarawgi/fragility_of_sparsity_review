import pandas as pd
import numpy as np
from src.normalization.drop_ref import process_categorical_numerical
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def model_fit_and_evaluate(x, D, y, model, model_name):
    """
    Fit a given model and evaluate its performance on the given data. 
    Also report the coefficients of the model.
    In particular, coefficients and standard errors for the treatment variable are reported.
    """
    