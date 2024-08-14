import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

def add_more_features(x, degree, case):
    """
    Add features to the data. we have 110 featues and we want it to increase it upto 1800
    """
    # original copy
    x_original = x.copy()

    # Convert x to DataFrame if it's not already
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)

    # 1. Polynomial Features (degree=2)
    poly = PolynomialFeatures(degree, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(x)
    poly_feature_names = poly.get_feature_names_out(x.columns)

    # 2. Interaction Features (degree=2 with interaction only)
    interaction = PolynomialFeatures(degree, interaction_only=True, include_bias=False)
    X_interactions = interaction.fit_transform(x)
    interaction_feature_names = interaction.get_feature_names_out(x.columns)

    # 3. Statistical Transformations: Log and Square Root
    X_log = np.log1p(x)
    X_sqrt = np.sqrt(x)
    log_feature_names = [f"log1p({col})" for col in x.columns]
    sqrt_feature_names = [f"sqrt({col})" for col in x.columns]

    # 4. Noise Addition
    noise = np.random.normal(0, 0.01, x.shape)
    X_noisy = x + noise
    noisy_feature_names = [f"noisy({col})" for col in x.columns]

    # Combining all generated features into a single matrix
    X_combined = np.hstack([X_poly, X_interactions, X_log, X_sqrt, X_noisy])
    combined_feature_names = np.hstack([poly_feature_names, interaction_feature_names, log_feature_names, sqrt_feature_names, noisy_feature_names])

    # shuffle 
    np.random.shuffle(X_combined)

    dimensions_a = int(0.90*x_original.shape[0])
    print("dimensions_a: ", dimensions_a)

    dimensions_b = int(1.10*x_original.shape[0])
    print("dimensions_b: ", dimensions_b)

    if X_combined.shape[1] > 1800:
        if case == 'close_to_n':
            X_combined = X_combined[:, :dimensions_a]
            combined_feature_names = combined_feature_names[:dimensions_a]
        elif case == 'more_than_n':
            X_combined = X_combined[:, :dimensions_b]
            combined_feature_names = combined_feature_names[:dimensions_a]
        else: 
            X_combined = x_original
            combined_feature_names = combined_feature_names[:dimensions_a]


    x_final = np.hstack([x_original, X_combined])
    final_feature_names = np.hstack([x_original.columns, combined_feature_names])
    # print("final_feature_names: ", list(final_feature_names))

    final_shape = x_final.shape
    print("final_shape: ", final_shape)

    if case == "more_than_n" or case == "close_to_n":
        print("Number of features added: ", final_shape[1] - x_original.shape[1])
        print("all features names: ", poly.get_feature_names_out())

    return x_final

# if __name__=='__main__':
#     x = np.random.rand(1900, 110)
#     print("Original shape of the data: ", x.shape)
#     add_more_features(x, degree=2, case='close_to_n')
#     add_more_features(x, degree=3, case='more_than_n')
#     add_more_features(x, degree=2, case='original')

#     print("Done!")
