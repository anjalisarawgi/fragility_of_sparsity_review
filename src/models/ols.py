import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import Lasso

def ols(x, y):
    X = sm.add_constant(x) # intercept 
    model = sm.OLS(y, X).fit()
    y_pred = model.predict(X)

    print("OLS Model Results............................................\n")
    print(model.summary())
    print("mean_squared_error",mean_squared_error(y, model.predict(X)))
    print("mean_absolute_error",mean_absolute_error(y, model.predict(X)))
    residuals = y - y_pred
    n = len(y)
    p = X.shape[1] - 1  # Subtract 1 to exclude the constant term
    standard_error = np.sqrt(np.sum(residuals**2) / (n - p - 1))
    print('Standard Error of the Model:', standard_error)

if __name__=='__main__':
    data = pd.read_csv('Data/screen_time/yoga.csv')
    print(data.head())

    data.columns = [col.strip() for col in data.columns]
    print(data.columns)
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%y')

    data['DayOfWeek'] = data['Date'].dt.dayofweek

    x = data[['Yoga', 'DayOfWeek', 'Social Networking', 'Reading and Reference','Other', 'Productivity', 'Health and Fitness','Entertainment','Creativity']]
    y = data['Total Screen Time']

    ols(x, y)
    print("OLS Model has been run successfully.")

