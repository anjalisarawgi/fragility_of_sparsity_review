import numpy as np
from scipy.stats import norm


def residual_test(rss_sbe, rss_ols, n, p):
    """Perform the Residual test."""

    # n, p = ols_model.df_resid + ols_model.df_model + 1, ols_model.df_model + 1
    
    # Calculate the F-statistic
    rss_diff = rss_ols - rss_sbe
    if rss_diff <= 0:
        print("rss_diff is not positive")
        return np.nan, np.nan  # Skip if the difference is not positive
    
    ##### not very sure
    
    f_stat = (rss_diff / p) / (rss_sbe / n)
    p_value = 1 - norm.cdf(np.sqrt(f_stat))
    
    return f_stat, p_value