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
from sklearn.linear_model import Lasso

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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = Lasso(alpha=0.1)
lasso_model= lasso.fit(X_scaled, y)
print("lasso_model",lasso_model)
y_pred = lasso.predict(X_scaled)
print('Mean Squared Error:', mean_squared_error(y, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y, y_pred))
print('lasso coefficients:', lasso.coef_)
print('lasso intercept:', lasso.intercept_)