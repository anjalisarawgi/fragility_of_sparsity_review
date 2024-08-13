import pandas as pd

def process_categorical_numerical(data, ref_cat_col=1):
    """Process categorical and numerical variables, and return dummified data with a specified reference category dropped."""
    categorical = data.select_dtypes(include=['object'])
    numerical = data.select_dtypes(include=['int64', 'float64'])
    
    print("Number of categorical variables: ", len(categorical.columns))
    print("Number of numerical variables: ", len(numerical.columns))
    
    print("Categorical variables: ", categorical.columns)
    
    # Convert categorical variables to dummy variables without dropping any category by default
    data_dummified = pd.get_dummies(data, drop_first=False)
    
    for col in categorical.columns:
        # Get all dummy columns for the categorical variable
        dummy_columns = [c for c in data_dummified.columns if c.startswith(col + '_')]
        
        if len(dummy_columns) >= ref_cat_col:
            # Determine which category to drop
            category_to_drop = dummy_columns[ref_cat_col - 1]
            data_dummified.drop(columns=[category_to_drop], inplace=True)
    
    print(f"Data shape after converting categorical variables and dropping the {ref_cat_col}th reference category: ", data_dummified.shape)
    
    return data_dummified
