import numpy as np
from scipy.stats import f, norm

def residual_test(rss_sbe, rss_ols, n, p):
    """Perform the Residual test to compare OLS and Sparsity-Based Estimator (SBE)."""
    
    # Compute the difference in RSS
    rss_diff = rss_sbe - rss_ols
    if rss_diff <= 0:
        print("RSS difference is not positive; residual test not valid.")
        return np.nan, np.nan  # Skip if the difference is not positive
    
    f_stat = (rss_diff / p) / (rss_sbe / (n - p))
    p_value = 1 - f.cdf(f_stat, p, n - p)
    
    return f_stat, p_value