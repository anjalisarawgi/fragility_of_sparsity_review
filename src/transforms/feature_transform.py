import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def add_more_features(x, degree, case):
    """
    Add features to the data. we have 110 featues and we want it to increase it upto 1800
    """
    # original copy
    x_original = x.copy()

    # 1. Polynomial Features (degree=2)
    poly = PolynomialFeatures(degree, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(x)

    # 2. Interaction Features (degree=2 with interaction only)
    interaction = PolynomialFeatures(degree, interaction_only=True, include_bias=False)
    X_interactions = interaction.fit_transform(x)

    # 3. Statistical Transformations: Log and Square Root
    X_log = np.log1p(x)
    X_sqrt = np.sqrt(x)

    # 4. Noise Addition
    noise = np.random.normal(0, 0.01, x.shape)
    X_noisy = x + noise

    # Combining all generated features into a single matrix
    X_combined = np.hstack([X_poly, X_interactions, X_log, X_sqrt, X_noisy])

    # shuffle 
    np.random.shuffle(X_combined)

    dimensions_a = int(0.90*x_original.shape[0])
    print("dimensions_a: ", dimensions_a)

    dimensions_b = int(1.10*x_original.shape[0])
    print("dimensions_b: ", dimensions_b)

    if X_combined.shape[1] > 1800:
        if case == 'close_to_n':
            X_combined = X_combined[:, :dimensions_a]
        elif case == 'more_than_n':
            X_combined = X_combined[:, :dimensions_b]
        else: 
            X_combined = x_original

    # combine the original features with the new features
    X_combined = np.hstack([x_original, X_combined])

    final_shape = X_combined.shape
    print(f"Final shape of the data after adding more features: {final_shape}")

    print("data types: ", X_combined.ctypes)
    return X_combined
