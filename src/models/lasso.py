import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
# import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import Lasso

def lasso(x, y, alpha):
    lasso = Lasso(alpha=0.1)
    lasso_model= lasso.fit(x, y)
    y_pred = lasso.predict(x)
    print("Lasso Model Results............................................\n")
    print('Coefficients:', lasso_model.coef_)
    print('Mean Squared Error:', mean_squared_error(y, y_pred))
    print('Mean Absolute Error:', mean_absolute_error(y, y_pred))

    # residuals
    residuals = y - y_pred
    n = len(y)
    p = x.shape[1]
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

    lasso(x, y, 0.1)
    print("Lasso Model has been run successfully.")
