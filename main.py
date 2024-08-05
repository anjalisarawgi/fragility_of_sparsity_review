import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.utils import resample

data = pd.read_csv('Data/yoga.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')

print(data.head())

# describe the data
print(data.isnull().sum())
print(data.describe())
print(data.columns  )
data.columns = [col.strip() for col in data.columns]
print(data.columns)
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')

data['DayOfWeek'] = data['Date'].dt.dayofweek

X = data[['Yoga', 'DayOfWeek', 'Social Networking', 'Reading and Reference','Other', 'Productivity', 'Health and Fitness','Entertainment','Creativity']]
y = data['Total Screen Time']

X = sm.add_constant(X) # intercept 

# OLS
model = sm.OLS(y, X).fit()
print(model.summary())
print("mean_squared_error",mean_squared_error(y, model.predict(X)))
print("mean_absolute_error",mean_absolute_error(y, model.predict(X)))

# Lasso
print("\nLasso Model............................................\n")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = Lasso(alpha=0.1)
lasso_model= lasso.fit(X_scaled, y)
print("lasso_model",lasso_model)
y_pred = lasso.predict(X_scaled)
print('Mean Squared Error:', mean_squared_error(y, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y, y_pred))

# selecting best alpha
lasso_cv = LassoCV(cv=10).fit(X_scaled, y)
best_alpha = lasso_cv.alpha_
# lasso_model = Lasso(alpha=best_alpha).fit(X_scaled, y)
print('Best alpha:', best_alpha)

# standard error for lasso using the best alpha -- se of the coefficients
def standard_error_lasso(X, y, bootstrap=1000, alpha=best_alpha):
    coefs = []
    for _ in range(bootstrap):
        X_resampled, y_resampled = resample(X, y)
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_resampled, y_resampled)
        coefs.append(lasso.coef_)
    coefs = np.array(coefs)
    coefs_mean = coefs.mean(axis=0)
    coefs_se = coefs.std(axis=0)
    return coefs_mean, coefs_se

coefs_mean, coefs_se = standard_error_lasso(X_scaled, y)
feature_names = ['Intercept'] + list(X.columns)
coef_summary = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': np.append(lasso_model.intercept_, coefs_mean),
    'Std_Error': np.append(np.nan, coefs_se),
})

print("coef_summary::", coef_summary)

# standard error for lasso using the best alpha -- se of the model
lasso_model = Lasso(alpha=best_alpha).fit(X_scaled, y)

residuals = y - lasso_model.predict(X_scaled)
residuals_std = np.std(residuals)
print("residuals_std",residuals_std)
# residuals_std_2 = residuals.std() # check if the residuals_std is the same as residuals.std()???
def standard_error_model(X, y, bootstrap=1000, alpha=best_alpha):
    residuals = []
    for _ in range(bootstrap):
        X_resampled, y_resampled = resample(X, y)
        lasso = Lasso(alpha=best_alpha)
        lasso.fit(X_resampled, y_resampled)
        resampled_residuals = y_resampled - lasso.predict(X_resampled)
        residuals.append(resampled_residuals)
    residuals = np.array(residuals)
    residuals_mean = residuals.mean(axis=0)
    residuals_std = residuals.std(axis=0)
    return residuals_mean, residuals_std

model_mean, model_se = standard_error_model(X_scaled, y)
print("model_mean",model_mean)
print("model_se",model_se)
