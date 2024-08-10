import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas as pd

def ols(x, y, feature_names):
    x = sm.add_constant(x) # intercept 
    result = sm.OLS(y, x).fit()
    print("ols model results:", result.summary())

