# import numpy as np
# from sklearn.linear_model import LassoCV
# import pickle
# import pandas as pd 


# def grid_search_alpha(x, y, alpha_file='models/communities_and_crime/optimal_alpha.pkl'):
#     # Define the range of alpha values to test
#     alphas = np.logspace(-4, 4, 20)
    
#     # Use LassoCV for automatic cross-validation to find the best alpha
#     lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
#     lasso_cv.fit(x, y)
    
#     best_alpha = lasso_cv.alpha_
#     print(f'Optimal alpha: {best_alpha}')
    
#     # Save the optimal alpha to a file
#     with open(alpha_file, 'wb') as file:
#         pickle.dump(best_alpha, file)
    
#     return best_alpha


# if __name__ == "__main__":
#     # Load the data
#     data = pd.read_csv('Data/communities_and_crime/processed/communities_and_crime.csv')
#     print('number of numerical columns: ', data.select_dtypes(include=['int64', 'float64']).shape[1])
#     print('number of categorical columns: ', data.select_dtypes(include=['category', 'object']).shape[1])
#     # convert the categorical columns to dummies
#     data = pd.get_dummies(data, drop_first=False)
#     print('number of columns after converting categorical columns to dummies: ', data.shape[1])
#     print("data.head(): ", data.head())
#     # Select the X and Y variables
#     x = data.drop(columns=['violentPerPop'])
#     y = data['violentPerPop']
    
#     # Perform grid search for the optimal alpha
#     best_alpha = grid_search_alpha(x, y)
#     print(f'Optimal alpha: {best_alpha}')