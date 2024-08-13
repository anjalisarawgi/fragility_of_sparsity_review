import pandas as pd

def process_categorical_numerical(data, ref_cat_col=1):
    """Process categorical and numerical variables, and return dummified data with a specified reference category dropped."""
    
    # Convert 'fold' to categorical
    data['fold'] = data['fold'].astype('category')
    
    # Drop 'communityname' column
    if 'communityname' in data.columns:
        data = data.drop(columns=['communityname'])  # Drop because it has too many unique values
    
    # Separate categorical and numerical columns after conversion
    categorical = data.select_dtypes(include=['category', 'object'])
    numerical = data.select_dtypes(include=['int64', 'float64'])
    
    print("Number of categorical variables: ", len(categorical.columns))
    print("Number of numerical variables: ", len(numerical.columns))
    
    # Convert categorical variables to dummies
    data_dummified = pd.get_dummies(data, drop_first=True)
    print(f"Data shape after converting categorical variables: ", data_dummified.shape)
    
    # Drop the specified reference category (if applicable)
    for col in categorical.columns:
        dummy_columns = [c for c in data_dummified.columns if c.startswith(col + '_')]
        if len(dummy_columns) >= ref_cat_col:
            category_to_drop = dummy_columns[ref_cat_col - 1]
            data_dummified.drop(columns=[category_to_drop], inplace=True)
    
    print(f"Data shape after dropping the {ref_cat_col}th reference category: ", data_dummified.shape)
    
    return data_dummified

if __name__ == '__main__':
    data = pd.read_csv('Data/processed/communities_and_crime.csv')
    print(data.head())
    print("data.shape: ", data.shape)
    data_dummified = process_categorical_numerical(data, ref_cat_col=1)
    print(data_dummified.head())
    print("data_dummified.shape: ", data_dummified.shape)
