import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold

def lasso(x, y, feature_names):
    params = {"alpha":np.arange(0.00001, 10, 500)}
    kf=KFold(n_splits=5,shuffle=True, random_state=42)

    lasso = Lasso()
    lasso_cv=GridSearchCV(lasso, param_grid=params, cv=kf)
    lasso_cv.fit(x, y)
    print("Best Params {}".format(lasso_cv.best_params_))

    # lasso model to plot the best features
    lasso_model = Lasso(alpha=lasso_cv.best_params_['alpha'])
    lasso_result = lasso_model.fit(x, y)
    print("lasso model results:", lasso_result.coef_)
    print("lasso model results:", lasso_result.intercept_)

    # Identify selected features
    selected_features = [feature_names[i] for i in range(len(feature_names)) if lasso_result.coef_[i] != 0]
    removed_features = [feature_names[i] for i in range(len(feature_names)) if lasso_result.coef_[i] == 0]

    print("Selected features after Lasso:", selected_features)
    print("Removed features after Lasso:", removed_features)



if __name__ == "__main__":
    feature_names = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
    ]
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    lasso(data, target, feature_names)