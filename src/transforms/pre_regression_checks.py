import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns

def check_multicollinearity(X):
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    print("\nVariance Inflation Factor (VIF):")
    print(vif_data)
    
    def interpret_vif(vif):
        if vif <= 1:
            return "No multicollinearity"
        elif vif > 1 and vif <= 5:
            return "Moderate multicollinearity"
        elif vif > 5 and vif <= 10:
            return "Significant multicollinearity"
        else:
            return "Severe multicollinearity"
        
    
    vif_data['Interpretation'] = vif_data['VIF'].apply(interpret_vif)
    print(vif_data)
    print("variables with VIF > 5: ", vif_data[vif_data['VIF'] > 5])

    return vif_data #, corr_matrix

def check_perfect_multicollinearity(X):
    duplicate_columns = X.T.duplicated().sum()
    if duplicate_columns > 0:
        print(f"\nWarning: {duplicate_columns} duplicate columns detected. Consider removing them to avoid perfect multicollinearity.")
        print("Duplicate columns:", X.columns[X.T.duplicated()])
    else:
        print("\nNo perfect multicollinearity detected (no duplicate columns).")

def check_linearity(X, y):
    # Scatter plots of each feature against the target variable
    for col in X.columns:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=X[col], y=y)
        plt.title(f'Scatter plot of {col} vs Target')
        plt.xlabel(col)
        plt.ylabel('Target')
        plt.savefig(f'src/reports/figures/scatter_{col}_vs_target.png')