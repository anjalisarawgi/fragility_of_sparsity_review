import pandas as pd
import numpy as np
from src.normalization.drop_ref import process_categorical_numerical
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import statsmodels.api as sm

def model_fit(x, D, y, model):
    if model == "Lasso":
        # step 1: Propensity score estimation
        lasso = LassoCV(cv=5).fit(x, D)
        selected_features = x.columns[lasso.coef_ != 0]
        print("number of selected features for the control matrix: ", len(selected_features))

        # step 2: outcome regression : ols  where y = D * Beta + W * Gamma + error 
        x_control = x[selected_features]

        # combine the treatment and control matrix
        X = pd.concat([D, x_control], axis=1)
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        print(model.summary())

    elif model == "post_double_lasso":
        pass
    elif model =="group_lasso":
        pass
    else:
        X = pd.concat([D, x], axis=1)
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        print(model.summary())

    return model

       

        





