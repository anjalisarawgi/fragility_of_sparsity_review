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
# from src.models.alpha import grid_search_alpha, find_optimal_alphas


def save_results(case, offset, model_name, ref_cat_col, dataset_name, 
                 features_selected_first_lasso, features_selected_second_lasso, 
                 features_selected_both_lasso,
                 treatment_coef, treatment_stderr, split_data, mse):
    """Save the results to a CSV file."""
    results_dir = os.path.join('results', dataset_name, case)
    os.makedirs(results_dir, exist_ok=True)
    results_dict = {
        'number of features selected (first lasso)': [len(features_selected_first_lasso)],
        'number of features selected (second lasso)': [len(features_selected_second_lasso)],
        'number of features selected (both lasso)': [len(features_selected_both_lasso)],
        'Treatment Coefficient': [treatment_coef],
        'Treatment StdErr': [treatment_stderr]
    }
    if mse is not None:
        results_dict['Mean Squared Error'] = [mse]
    
    results_df = pd.DataFrame(results_dict)
    split_status = "split" if split_data else "no_split"
    results_df.to_csv(os.path.join(results_dir, f'{model_name}_{offset}_{ref_cat_col}_{split_status}.csv'), index=False)


def train_and_evaluate_model(x, D, y, model_name, dataset_name, split_data = False):
    """
    Fit the model on the full dataset and return the selected features and treatment coefficient.
    """
    if split_data:
        X_D = pd.concat([x, D], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X_D, y, test_size=0.2, random_state=42)

        if dataset_name == 'communities_and_crime':
            x = X_train.drop(columns='pop')
            D = X_train['pop']
            y = y_train
        elif dataset_name == 'lalonde':
            x = X_train.drop(columns='treat')
            D = X_train['treat']
            y = y_train

        model, features_selected_first_lasso, features_selected_second_lasso, features_selected_both_lasso  = model_fit(x, D, y, model_name)

        # calculate the mean squared error on the test set
        X_test_selected = X_test[model.params.index[1:]]
        X_test_selected = sm.add_constant(X_test_selected, has_constant='add')
        y_pred = model.predict(X_test_selected)
        mse = mean_squared_error(y_test, y_pred)
    else:
        model, features_selected_first_lasso, features_selected_second_lasso, features_selected_both_lasso  = model_fit(x, D, y, model_name)
        mse = None

    treatment_coef = model.params[D.name]
    treatment_stderr = model.bse[D.name]
    
    return features_selected_first_lasso, features_selected_second_lasso, features_selected_both_lasso, treatment_coef, treatment_stderr, mse





def main(dataset_path, ref_cat_col, offset, model_name, case, split_data = False):
    # Load the data
    data = pd.read_csv(dataset_path)
    dataset_name = os.path.basename(dataset_path).replace('.csv', '')

    # handling categorical variables
    data, categorical_columns = process_categorical_numerical(data, dataset_name)
    data = drop_ref_cat(data, ref_cat_col, categorical_columns)

    # Select the X, Y, and D variables
    if dataset_name == 'communities_and_crime':
        x = data.drop(columns=['violentPerPop', 'pop'])
        D = data['pop']
        y = data['violentPerPop']
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
        if dataset_name == 'communities_and_crime':
            x = add_more_features(x, degree=2, case = case, dataset_name=dataset_name, seed=42)
        elif dataset_name == 'lalonde':
            x = add_more_features(x, degree=3, case = case, dataset_name=dataset_name, seed=42)
        print("Shape of x after adding more features: ", x.shape)
    
    x = normalize_data(x, offset)  # normalize the data 
    
    if dataset_name == 'lalonde':
        D = D.astype(int)

    # Fit the model on the entire dataset
    features_selected_first_lasso, features_selected_second_lasso,features_selected_both_lasso, treatment_coef, treatment_stderr, mse = train_and_evaluate_model(x, D, y, model_name, dataset_name, split_data)  

    save_results(case, offset, model_name, ref_cat_col, dataset_name, 
                 features_selected_first_lasso, features_selected_second_lasso, 
                 features_selected_both_lasso,
                 treatment_coef, treatment_stderr,split_data, mse)


if __name__ == '__main__':
    crime = 'Data/communities_and_crime/processed/communities_and_crime.csv'
    lalonde = "Data/lalonde/processed/lalonde.csv"
    main(dataset_path =lalonde ,  ref_cat_col=2, offset="demean", model_name='post_double_lasso', case='original', split_data=True)


# problem: lasso gives different results for different runs
    

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