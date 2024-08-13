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

def main(ref_cat_col):
    # load the data
    data = pd.read_csv('Data/processed/communities_and_crime.csv')
    print(data.head())
    y = data['ViolentCrimesPerPop']
    D = data['population']
    x = data.drop(columns=['ViolentCrimesPerPop', 'population'])

    print("number of categorical columns: ", len(data.select_dtypes(include=['object']).columns))

    
    # convert to dummies and drop the reference category
    print("number of x columns before dummification: ", len(data.columns)-2)
    data_dummified = process_categorical_numerical(data, ref_cat_col=ref_cat_col)

    print("number of columns after dummification: ", len(data_dummified.columns)-2)
    x = data_dummified.drop(columns=['ViolentCrimesPerPop', 'population'])
    print("x shape: ", x.shape, "D shape: ", D.shape, "y shape: ", y.shape)
   

    # model 
    model = model_fit(x, D, y, model="ols")
    print(model.summary())




if __name__ == '__main__':
    main(ref_cat_col=1)


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