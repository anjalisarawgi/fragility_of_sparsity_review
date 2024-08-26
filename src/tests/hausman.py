from scipy.stats import norm
import numpy as np
import pandas as pd
import statsmodels.api as sm


def hausman_test(ols_model, sbe_model):
    """
    Perform a Hausman test to compare OLS and Sparsity-Based Estimator (SBE).
    
    Parameters:
    ols_model : Fitted OLS model (statsmodels)
    sbe_model : Fitted SBE model (statsmodels)
    
    Returns:
    test_stat : Test statistic for the Hausman test
    p_value : p-value for the test
    """
    diff = ols_model.params - sbe_model.params
    var_diff = ols_model.bse**2 - sbe_model.bse**2
    test_stat = (diff**2 / var_diff).sum()
    p_value = 1 - norm.cdf(np.sqrt(test_stat))
    return test_stat, p_value
