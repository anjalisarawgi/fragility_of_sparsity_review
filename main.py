import pandas as pd
import numpy as np
import os
from src.normalization.drop_ref import process_categorical_numerical, drop_ref_cat
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from src.models.model import model_fit
import statsmodels.api as sm
from src.normalization.offsets import normalize_data
from src.transforms.feature_transform import add_more_features


def save_results(case, method,model, model_name, ref_cat_col, dataset_name):
    results_dir = os.path.join('results',dataset_name, case)
    os.makedirs(results_dir, exist_ok=True)
    coef_df = pd.DataFrame({
        'number of features selected': model.params.shape[0],
        'Coefficient': model.params,
        'StdErr': model.bse, 
        'R2': model.rsquared,
    })
    coef_df.index.name = 'Feature'
    coef_df.to_csv(os.path.join(results_dir, f'{model_name}_model_coefficients_{method}_{ref_cat_col}.csv'), index=True)


def train_and_evaluate_model(x, D, y, model_name, dataset_name):
    X_D = pd.concat([x, D], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_D, y, test_size=0.2, random_state=42)
    if dataset_name == 'communities_and_crime':
        x = X_train.drop(columns='population')
        D = X_train['population']
        y = y_train
    elif dataset_name == 'lalonde':
        x = X_train.drop(columns='treat')
        D = X_train['treat']
        y = y_train

    model = model_fit(x, D, y, model_name)
    X_test_selected = X_test[model.params.index[1:]]    
    X_test_selected = sm.add_constant(X_test_selected, has_constant='add')
    y_pred = model.predict(X_test_selected)
    mse = mean_squared_error(y_test, y_pred)
    return mse


def main(dataset_path, ref_cat_col, method, model_name, case):
    # Load the data
    data = pd.read_csv(dataset_path)
    dataset_name = os.path.basename(dataset_path).replace('.csv', '')

    # handling categorical variables
    data, categorical_columns = process_categorical_numerical(data, dataset_name)
    data = drop_ref_cat(data, ref_cat_col, categorical_columns)

    # Select the X, Y, and D variables
    if dataset_name == 'communities_and_crime':
        x = data.drop(columns=['ViolentCrimesPerPop', 'population'])
        D = data['population']
        y = data['ViolentCrimesPerPop']
    elif dataset_name == 'lalonde':
        x = data.drop(columns=['re78', 'treat'])
        D = data['treat']
        y = data['re78']
    
    # feature engineering
    if case == "original":
        print("Original case: Using the original features.")
        print("HEAD of x: ", x.head())
    elif case == "close_to_n" or case == "more_than_n":
        print("Adding more features.")
        x = add_more_features(x, degree=2, case = case)
        print("Shape of x after adding more features: ", x.shape)
    
    x = normalize_data(x, method)  # normalize the data 
    # print("x.head() after normalization: ", x.head())
    
    if dataset_name == 'lalonde':
        D = D.astype(int)

    # model fitting
    model = model_fit(x, D, y, model_name)
    print(model.summary())
    if dataset_name == 'communities_and_crime':
        print("population coef and std err: ", model.params['population'], model.bse['population'])
    elif dataset_name == 'lalonde':
        print("treat coef and std err: ", model.params['treat'], model.bse['treat'])


    # mse = train_and_evaluate_model(x, D, y, model_name, dataset_name)  
    save_results(case, method, model,model_name, ref_cat_col, dataset_name)



if __name__ == '__main__':
    crime = 'Data/communities_and_crime/processed/communities_and_crime.csv'
    lalonde = "Data/lalonde/processed/lalonde.csv"
    main(dataset_path =lalonde ,  ref_cat_col=2, method="demean", model_name='post_double_lasso', case='original')


# convert
### keep the main set intact after kepeing interactions
    
## woah the mean normalization is very very stable -- include this in the report (median is not)

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