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
# from src.models.alpha import grid_search_alpha



# def load_optimal_alpha(alpha_file='models/communities_and_crime/optimal_alpha.pkl'):
#     with open(alpha_file, 'rb') as file:
#         return pickle.load(file)
    
def model_fit(x, D, y, model):
    print("x shape: ", x.shape)
    print("D shape: ", D.shape)
    print("check if x has na or inf values: ", x.isnull().values.any())
    # check how many na values in x
    print("number of na values in x: ", x.isnull().sum().sum())
    x = x.astype(int)

    selected_features_D = []
    selected_features_Y = []
    selected_features = []
    
    if model == "post_double_lasso":
        print("total number of features: ", x.shape[1])

        # Load the optimal alpha if provided
        # if alpha is None:
        #     alpha = load_optimal_alpha()
        #     print(f'Loaded optimal alpha: {alpha}')
        # first lasso: regress D on covariates 
        lasso_D = Lasso().fit(x, D)
        # lasso_D = LassoCV(cv=5, random_state=42).fit(x, D)
        selected_features_D = x.columns[lasso_D.coef_ != 0]
        print("selected features from first lasso: ", len(selected_features_D))

        # second lasso: regress y on covariates
        lasso_Y = Lasso().fit(x, y)
        # lasso_Y = LassoCV(cv=5, random_state=42).fit(x, y)
        selected_features_Y = x.columns[lasso_Y.coef_ != 0]
        print("selected features from second lasso: ", len(selected_features_Y))

        # union of the selected features
        selected_features = selected_features_D.union(selected_features_Y)
        print("Selected features from both lasso models:", len(selected_features))

        # finally, OLS regression with the selected features
        x_control = x[selected_features]
        X = pd.concat([D, x_control], axis=1)
        X = sm.add_constant(X)
        sbe_model = sm.OLS(y, X).fit()

        # 2 : ols on all the features
        ols_model = sm.OLS(y, sm.add_constant(pd.concat([D, x], axis=1))).fit() 

    
    # elif model == "neural_network":
    #     X = pd.concat([D, x], axis=1)
    #     mlp = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
    #     mlp.fit(X, y)

    else: 
        X = pd.concat([D, x], axis=1)
        X = sm.add_constant(X)
        ols_model = sm.OLS(y, X).fit()
        sbe_model = ols_model

    return sbe_model, ols_model, selected_features_D, selected_features_Y, selected_features
       

    