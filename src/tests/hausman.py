from scipy.stats import norm
import numpy as np
import pandas as pd
import statsmodels.api as sm

def hausman_test(treatment_coeff_ols, treatment_coeff_sbe, treatment_stderr_ols, treatment_stderr_sbe):
    """
    Perform a Hausman test to compare OLS and Sparsity-Based Estimator (SBE).
    
    Parameters:
    treatment_coeff_ols : Coefficient estimate from OLS
    treatment_coeff_sbe : Coefficient estimate from SBE
    treatment_stderr_ols : Standard error of OLS coefficient
    treatment_stderr_sbe : Standard error of SBE coefficient
    
    Returns:
    test_stat : Test statistic for the Hausman test
    p_value : p-value for the test
    """

    diff = treatment_coeff_ols - treatment_coeff_sbe
    var_diff = treatment_stderr_ols**2 - treatment_stderr_sbe**2

    if var_diff < 0:
        var_diff = -var_diff # because we are interested in the magnitude


    test_stat = (diff**2 / var_diff).sum()
    p_value = 1 - norm.cdf(np.sqrt(test_stat))

    return test_stat, p_value