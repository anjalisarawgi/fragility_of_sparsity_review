import pandas as pd
import numpy as np
from normalization.drop_ref import process_categorical_numerical
from sklearn.linear_model import LassoCV
import statsmodels.api as sm

def model_fit(x, D, y, model="ols"):
    # Convert all columns to numeric, coercing errors to NaN and dropping them
    x = x.apply(pd.to_numeric, errors='coerce')
    D = pd.to_numeric(D, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    x = x.astype(int)
    
    # Drop any remaining NaNs after conversion
    x = x.dropna()
    D = D.dropna()
    y = y.dropna()
    
    if model == "Lasso":
        # Step 1: Propensity score estimation with Lasso
        lasso = LassoCV(cv=5).fit(x, D)
        selected_features = x.columns[lasso.coef_ != 0]
        print("Number of selected features for the control matrix: ", len(selected_features))

        # Step 2: Outcome regression with OLS
        x_control = x[selected_features]
        X = pd.concat([D, x_control], axis=1)  # Combine the treatment and control matrix
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        print(model.summary())

    elif model == "post_double_lasso":
        pass  # Implement the Post-Double Lasso procedure here

    elif model == "grouped_lasso":
        pass  # Implement the Grouped Lasso procedure here

    elif model == "ols":
        # Basic OLS model
        X = pd.concat([D, x], axis=1)
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        print(model.summary())

    else:
        raise ValueError(f"Model '{model}' not recognized. Please choose from 'ols', 'Lasso', 'post_double_lasso', or 'grouped_lasso'.")

    return model

if __name__ == '__main__':
    data = pd.read_csv('Data/processed/communities_and_crime.csv')

    # Process the data by converting categorical variables and dummifying
    data_dummified = process_categorical_numerical(data, ref_cat_col=1)
    print(data_dummified.head())
    print("data_dummified.shape: ", data_dummified.shape)

    # Define the outcome, treatment, and predictors
    y = data_dummified['ViolentCrimesPerPop']
    D = data_dummified['population']
    x = data_dummified.drop(columns=['ViolentCrimesPerPop', 'population'])

    # Fit the model
    model = model_fit(x, D, y, model="Lasso")
    print(model.summary())