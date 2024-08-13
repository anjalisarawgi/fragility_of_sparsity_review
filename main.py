import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import os
from src.normalization.drop_ref import process_categorical_numerical
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from src.model import model_fit
import statsmodels.api as sm
from src.normalization.offsets import normalize_data

def main(ref_cat_col):
    # load the data
    data = pd.read_csv('Data/processed/communities_and_crime.csv')
    
    print("number of categorical columns: ", len(data.select_dtypes(include=['object']).columns))
    
    print("number of x columns before dummification: ", len(data.columns)-2)
    data_dummified = process_categorical_numerical(data, ref_cat_col=ref_cat_col)
    print("number of columns after dummification: ", len(data_dummified.columns)-2)
    
    x = data_dummified.drop(columns=['ViolentCrimesPerPop', 'population'])
    D = data_dummified['population']
    y = data_dummified['ViolentCrimesPerPop']

    print("x shape: ", x.shape, "D shape: ", D.shape, "y shape: ", y.shape)
   
    # normalize the data
    x = normalize_data(x, method="demean")  


    # model 
    model = model_fit(x, D, y, model="lasso")
    # print(model.summary())
    print("population coef and std err: ", model.params['population'], model.bse['population'])

    

    # train test split
    # Combine X and D into a single feature set
    X_D = pd.concat([x, D], axis=1)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_D, y, test_size=0.2, random_state=42)

    print("Training set size: ", X_train.shape)
    print("Test set size: ", X_test.shape)
    print("y_train size: ", y_train.shape)

    # Fit the model on the training data
    x = X_train.drop(columns='population')
    D = X_train['population']
    y = y_train
    model = model_fit(x, D, y, model="lasso")
    print("number of features selected: ", len(model.params)-1)
    # print(model.summary())
    print("x.shape: ", x.shape, "D.shape: ", D.shape, "y.shape: ", y.shape)
    print("X_test shape: ", X_test.shape)

    print("selected features: ", model.params.index)

    
    # X_test =  X_test[model.params.index[1:]].shape

    # remove constant from the model
    print("xtest with the selected features: ",X_test[model.params.index[1:]].shape)

    selected_features = model.params.index
    print("selected features: ", selected_features)
   
    # y_pred = model.predict( X_test[model.params.index[1:]].shape)
    # print("y_pred shape: ", y_pred.shape)

    X_test_selected = X_test[model.params.index[1:]]    
    X_test_selected = sm.add_constant(X_test_selected, has_constant='add')
    print("X_test_selected shape: ", X_test_selected.shape)

    y_pred = model.predict(X_test_selected)


    # Evaluate the model performance
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error on the test set: ", mse) # 0.023750639979623824





if __name__ == '__main__':
    main(ref_cat_col= 3) 


# convert
 


######## check for outliers

# check for multicollinearity

# check for normaliity of the data


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

# case 1: p as same : 
# a. ols, lasso, post lasso, double lasso 
# b. tests for p

# case 2: p as high 
# a. ols, lasso, post lasso, double lasso 
# b. tests for p

# case 3: p as more than n 
# a. ols, lasso, post lasso, double lasso 
# b. tests for p

# adding noisy features too 
# adding additonal multicollinear features too

# model selection vs lasso 
# aic bic 
# cross validation
# try different lambdas 
# try different sparsity values