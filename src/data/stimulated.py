import numpy as np
import pandas as pd

# Parameters
n = 100  # number of observations
p = 150  # number of control variables (p > n to mimic high-dimensional setting)
sparsity = 10  # number of non-zero coefficients (sparse setting)

# Generate control variables (W)
np.random.seed(42)
W = np.random.randn(n, p)

# Introduce multicollinearity by adding linear combinations of some variables
W[:, 0] = W[:, 1] + W[:, 2] + np.random.normal(0, 0.01, n)

# Generate a sparse coefficient vector for control variables
beta = np.zeros(p)
non_zero_indices = np.random.choice(np.arange(p), sparsity, replace=False)
beta[non_zero_indices] = np.random.randn(sparsity)

# Generate the treatment variable (D)
D = W[:, :sparsity].dot(np.random.randn(sparsity)) + np.random.normal(0, 1, n)

# Generate the outcome variable (Y) as a linear combination of D and W
Y = D * 2.5 + W.dot(beta) + np.random.normal(0, 1, n)

# Create a DataFrame
df = pd.DataFrame(W, columns=[f"W{i+1}" for i in range(p)])
df['D'] = D
df['Y'] = Y

# Display the first few rows of the dataset
print(df.head())

