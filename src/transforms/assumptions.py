import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro, probplot

def check_assumptions_after(X, y, model):
    """ Function to check the assumptions of linear regression """
    
    # 1. Linearity
    # Predicted vs. Actual Plot
    y_pred = model.predict(X)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, y)
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Predicted vs. Actual Values')
    plt.savefig('src/reports/figures/Predicted_vs_Actual.png')

    # Residuals vs. Predicted Values
    residuals = y - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted Values')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.savefig('src/reports/figures/Residuals_vs_Predicted.png')
    
    # 2. Independence
    # Durbin-Watson test
    dw_stat = durbin_watson(residuals)
    print(f'Durbin-Watson statistic: {dw_stat:.3f}')
    
    # 3. Homoscedasticity
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Checking Homoscedasticity (Residuals vs. Predicted Values)')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.savefig('src/reports/figures/Homoscedasticity.png')
    
    # 4. Normality of Residuals
    # Histogram of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Histogram of Residuals')
    plt.savefig('src/reports/figures/Histogram_of_Residuals.png')
    
    # Q-Q plot
    plt.figure(figsize=(10, 6))
    probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.savefig('src/reports/figures/QQ_Plot_of_Residuals.png')
    
    # Shapiro-Wilk test for normality
    shapiro_test = shapiro(residuals)
    print(f'Shapiro-Wilk test p-value: {shapiro_test.pvalue:.3f}')
    if shapiro_test.pvalue > 0.05:
        print("Residuals are normally distributed (fail to reject H0).")
    else:
        print("Residuals are not normally distributed (reject H0).")

