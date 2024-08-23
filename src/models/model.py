import pandas as pd
import numpy as np
# from src.normalization.drop_ref import process_categorical_numerical
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import statsmodels.api as sm
from causalinference import CausalModel

def model_fit(x, D, y, model):

    x = x.astype(int)

    if model == "post_double_lasso":
        print("total number of features: ", x.shape[1])
        # first lasso: regress D on covariates 
        lasso_D = Lasso(alpha=0.01).fit(x, D)
        selected_features_D = x.columns[lasso_D.coef_ != 0]
        print("selected features from first lasso: ", len(selected_features_D))

        # second lasso: regress y on covariates
        lasso_Y = LassoCV(cv=5).fit(x, y)
        selected_features_Y = x.columns[lasso_Y.coef_ != 0]
        print("selected features from second lasso: ", len(selected_features_Y))

        # union of the selected features
        selected_features = selected_features_D.union(selected_features_Y)
        print("Selected features from both lasso models:", len(selected_features))

        # finally, OLS regression with the selected features
        x_control = x[selected_features]
        X = pd.concat([D, x_control], axis=1)
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
    
    elif model == "causal_model_lasso":
        # first lasso: regress D on covariates 
        lasso_D = Lasso(alpha=0.01).fit(x, D)
        selected_features_D = x.columns[lasso_D.coef_ != 0]
        print("selected features from first lasso: ", len(selected_features_D))

        # causal model with the selected features
        x_control = x[selected_features_D]
        causal = CausalModel(Y=y.values, D=D.values, X=x_control.values)
        causal.est_via_ols()
        print(causal.estimates)
        print(causal.summary_stats)


        

        

    else: 
        X = pd.concat([D, x], axis=1)
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
    return model
       

    