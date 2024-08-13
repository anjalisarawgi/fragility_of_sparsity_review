import pandas as pd
import numpy as np
from src.normalization.drop_ref import process_categorical_numerical
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import statsmodels.api as sm

def model_fit(x, D, y, model):
    # Convert all columns to numeric, coercing errors to NaN and dropping them
    x = x.apply(pd.to_numeric, errors='coerce')
    D = pd.to_numeric(D, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    
    # Convert boolean dummies to integers (0, 1)
    x = x.astype(int)
    
    # Drop any remaining NaNs after conversion
    x = x.dropna()
    D = D.dropna()
    y = y.dropna()

    if model == "lasso":
        # step 1: Propensity score estimation
        lasso = LassoCV(cv=5).fit(x, D)
        # lasso = Lasso(alpha=0.1).fit(x, D)
        selected_features = x.columns[lasso.coef_ != 0]
        print("number of selected features for the control matrix: ", len(selected_features))
        print("selected features: ", selected_features)
        print("alpha: ", lasso.alpha_)

        # step 2: outcome regression : ols  where y = D * Beta + W * Gamma + error 
        x_control = x[selected_features]
        print("x_control shape: ", x_control.shape)
        X = pd.concat([D, x_control], axis=1)  # combine the treatment and control matrix
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        # print(model.summary())

    elif model == "post_double_lasso":
        pass
    elif model =="grouped_lasso":
        pass

    else: 
        X = pd.concat([D, x], axis=1)
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
    return model
       

        
if __name__ == '__main__':
    data = pd.read_csv('Data/processed/communities_and_crime.csv')
   
    data_dummified = process_categorical_numerical(data, ref_cat_col=1)
    print(data_dummified.head())
    print("data_dummified.shape: ", data_dummified.shape)
    y = data_dummified['ViolentCrimesPerPop']
    D = data_dummified['population']
    x = data_dummified.drop(columns=['ViolentCrimesPerPop', 'population'])
    model = model_fit(x, D, y, model="lasso")
    print(model.summary())
