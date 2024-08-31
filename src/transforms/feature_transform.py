import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

def add_more_features(x, degree, case, dataset_name, seed):
    np.random.seed(seed)
    x_original = x.copy()

    # Convert x to DataFrame if it's not already
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)

    #Identify numeric and categorical columns
    # numeric_cols = x.select_dtypes(include=[np.number]).columns
    # categorical_cols = x.select_dtypes(include=['object', 'category']).columns

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

    dimensions_a = int(0.80*x_original.shape[0])
    dimensions_b = int(1.10*x_original.shape[0])

    if dataset_name == 'communities_and_crime':
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
    elif dataset_name == 'lalonde':
        if X_combined.shape[1] > 400:
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
    x_final = pd.DataFrame(x_final)
    
    x_final = x_final.apply(pd.to_numeric, errors='ignore')

    return x_final
