import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import os
from src.normalization.drop_ref import process_categorical_numerical
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from src.models.model import model_fit
import statsmodels.api as sm
from src.normalization.offsets import normalize_data
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures
from src.transforms.feature_transform import add_more_features
from src.transforms.assumptions import check_assumptions_after
from src.transforms.pre_regression_checks import check_multicollinearity, check_perfect_multicollinearity, check_linearity

def save_results(case, method,model, model_name, mse, ref_cat_col):
    """ Save results to the 'results' directory. """

    results_dir = os.path.join('results', case)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save model coefficients and standard errors
    coef_df = pd.DataFrame({
        'number of features selected': model.params.shape[0],
        'Coefficient': model.params,
        'StdErr': model.bse, 
        'R2': model.rsquared,
        'MSE': mse, 
    })

    # Set the feature names as the index
    coef_df.index.name = 'Feature'

    # Save the DataFrame to a CSV file
    coef_df.to_csv(os.path.join(results_dir, f'{model_name}_model_coefficients_{method}_{ref_cat_col}.csv'), index=True)

    # coef_df.to_csv(os.path.join(results_dir, f'{model_name}_model_coefficients_{method}_{ref_cat_col}.csv'), index=True)


def train_and_evaluate_model(x, D, y, model_name):
    X_D = pd.concat([x, D], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_D, y, test_size=0.2, random_state=42)

    x = X_train.drop(columns='population')
    D = X_train['population']
    y = y_train

    model = model_fit(x, D, y, model_name) # model on the training data

    # Select the features in the test set based on the model
    X_test_selected = X_test[model.params.index[1:]]    
    X_test_selected = sm.add_constant(X_test_selected, has_constant='add')
    print("X_test_selected shape: ", X_test_selected.shape)

    y_pred = model.predict(X_test_selected)
    # Evaluate the model performance
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error on the test set: ", mse) 

    return mse





def main(ref_cat_col, method, model_name, case):
    # load the data
    data = pd.read_csv('Data/processed/communities_and_crime.csv')
    data = process_categorical_numerical(data, ref_cat_col=ref_cat_col)
    
    x = data.drop(columns=['ViolentCrimesPerPop', 'population'])
    D = data['population']
    y = data['ViolentCrimesPerPop']

    if case == "original":
        print("Original case: Using the original features.")
    elif case == "close_to_n" or case == "more_than_n":
        print("Adding more features.")
        x = add_more_features(x, degree=2, case = case)  # Interaction terms limited to around 1800 features
        print("Shape of x after adding more features: ", x.shape)

    print(f"Shape of x after feature engineering ({case}): ", x.shape)


    x = pd.DataFrame(x)
    
    x = x.astype(int) # convert boolean dummies to integers (0, 1)
    print(" data types: ", x.dtypes.value_counts())

    # # 1. Multicollinearity Check
    # vif_data = check_multicollinearity(x)
    # columns_to_drop = ['PctRecImmig5', 'PctRecImmig8', 'PctRecImmig10', 'PctLargHouseFam', 'PctLargHouseOccup', 'PersPerOccupHous', 'OwnOccMedVal', 'PctSameHouse85', 'perCapInc', 'whitePerCap']
    # x_dropped = x.drop(columns=columns_to_drop)

    # vif_data = check_multicollinearity(x_dropped)

    # print("Variance Inflation Factor (VIF):", vif_data)

    # # 2. Perfect Multicollinearity Check
    # check_perfect_multicollinearity(x_dropped)

    # # 3. Linearity Check
    # check_linearity(x_dropped, y)



    x = normalize_data(x, method)  # normalize the data

    model = model_fit(x, D, y, model_name) # model
    print("population coef and std err: ", model.params['population'], model.bse['population'])


    # Fit the model on the training data and evaluate on the test data
    mse = train_and_evaluate_model(x, D, y, model_name)  
    # x = x.unsqueeze(0)
    # check_assumptions_after(x, y, model)

    # save the results
    save_results(case, method, model,model_name,  mse, ref_cat_col)


if __name__ == '__main__':
    main(ref_cat_col= 1, method='demean', model_name='lasso', case='more_than_n') 


# convert
 

### keep the main set intact after kepeing interactions
    
    # there is a problem with increasing the dimensions of the data
    # it isnt working as expected
######## check for outliers

# check for multicollinearity
# autocorrelation
# heterogeneity
# check for normaliity of the data
# vif and saving results are left

# preprocessing 
# outliers
# feature selection
# feature scaling
# encoding categorical variables
# splitting the data into training and testing sets

# selecting the treatment and outcome variables
# treatment variable

# outcome variable

# assumptions to take care of 

# normalization methods: demenaning, median substraction, minmax scaling maybe, dropping collinear columns, subsets of columns drops, 

# pls check / convert to categorical variables  -- but is this behaviour really really unexpected?
    
# case 1: p as same : (dont forget to increase the dimensions)
# a. ols, lasso, post lasso, double lasso 
# b. tests for p
# c. all methods of normalization

# case 2: p as high -- also have train test split for this  (dont forget to increase the dimensions)
# a. ols, lasso, post lasso, double lasso 
# b. tests for p
# c. all methods of normalization


# case 3: p as more than n (dont forget to increase the dimensions)
# a. ols, lasso, post lasso, double lasso 
# b. tests for p
# c. all methods of normalization

# adding noisy features too 
# adding additonal multicollinear features too

# model selection vs lasso 
# aic bic 
# cross validation
# try different lambdas 
# try different sparsity values