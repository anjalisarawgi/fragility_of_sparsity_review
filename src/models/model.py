import pandas as pd
import numpy as np
# from src.normalization.drop_ref import process_categorical_numerical
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import statsmodels.api as sm
from causalinference import CausalModel
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
import pickle
    
def model_fit(x, D, y, model, dataset_name):
    x = x.astype(int)

    selected_features_D = []
    selected_features_Y = []
    selected_features = []
    
    if model == "post_double_lasso":
        ols_model = sm.OLS(y, sm.add_constant(pd.concat([D, x], axis=1))).fit() 

        # first lasso
        if dataset_name == 'communities_and_crime':
            lasso_D = Lasso(alpha=10, random_state=42).fit(x, D)
            # lasso_D = LassoCV(cv=5, random_state=42).fit(x, D)
        elif dataset_name == 'lalonde': 
            lasso_D = Lasso(random_state=42).fit(x, D)
            # lasso_D = LassoCV(cv=5, random_state=42).fit(x, D)
        selected_features_D = x.columns[lasso_D.coef_ != 0]

        # second lasso
        if dataset_name == 'communities_and_crime':
            lasso_Y = Lasso(alpha=10, random_state=42).fit(x, y)
             # lasso_Y = LassoCV(cv=5, random_state=42).fit(x, y)
        elif dataset_name == 'lalonde': 
            lasso_Y = Lasso(random_state=42).fit(x, y)
            # lasso_Y = LassoCV(cv=5, random_state=42).fit(x, y)
    
        selected_features_Y = x.columns[lasso_Y.coef_ != 0]
        selected_features = selected_features_D.union(selected_features_Y)

        # ols
        x_control = x[selected_features]
        X = pd.concat([D, x_control], axis=1)
        X = sm.add_constant(X)
        sbe_model = sm.OLS(y, X).fit()

    else: # if model == "ols"
        X = pd.concat([D, x], axis=1)
        X = sm.add_constant(X)
        ols_model = sm.OLS(y, X).fit()
        sbe_model = ols_model

    return sbe_model, ols_model, selected_features_D, selected_features_Y, selected_features
       

    