import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
# import statsmodels.api as sm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.utils import resample
from src.models.lasso import lasso
from src.models.ols import ols
from sklearn.preprocessing import OneHotEncoder

def main():
    train_data = pd.read_csv('Data/MNIST/mnist_train.csv')
    test_data = pd.read_csv('Data/MNIST/mnist_test.csv')

    # features and labels (60000, 784) (60000,)
    x_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    print(x_train.shape, y_train.shape)

    x_test = test_data.drop('label', axis=1)
    y_test = test_data['label']

    # one hot encoding
    encoder =   OneHotEncoder()
    y_train_onehot = encoder.fit_transform(y_train.values.reshape(-1,1)).toarray()
    y_test_onehot = encoder.transform(y_test.values.reshape(-1,1)).toarray()

    lasso = Lasso(alpha=0.1)
    lasso_model= lasso.fit(x_train, y_train_onehot)
    y_pred = lasso.predict(x_test)

    y_pred_labels = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred_labels == y_test)
    print("Accuracy of the Lasso Model:", accuracy)

    print("lasso model results:")
    print('Coefficients:', lasso_model.coef_)
    print('Mean Squared Error:', mean_squared_error(y_test_onehot, y_pred))


    


if __name__=='__main__':
    main()
    print("Model has been run successfully.")