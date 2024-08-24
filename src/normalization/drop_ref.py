import pandas as pd

def process_categorical_numerical(data, dataset_name):  
    """Process categorical and numerical variables, and return dummified data with a specified reference category dropped."""
    
    if dataset_name == 'communities_and_crime_unorm':
        print("categories: ", data.select_dtypes(include=['category', 'object']).columns)
        data_dummified = pd.get_dummies(data, drop_first=False) ###???
        categorical = data.select_dtypes(include=['category', 'object'])
        numerical = data.select_dtypes(include=['int64', 'float64'])

    elif dataset_name == 'lalonde':
        columns_to_convert = ['black', 'hisp', 'married', 'nodegr']
        data[columns_to_convert] = data[columns_to_convert].astype(object)
        data_dummified = pd.get_dummies(data, drop_first=False)
        categorical = data.select_dtypes(include=['category', 'object', 'bool'])
        numerical = data.select_dtypes(include=['int64', 'float64'])

    return data_dummified, categorical.columns

    

def drop_ref_cat(data_dummified, ref_cat_col, categorical_columns):
    # Drop the specified reference category (if applicable)
    for col in categorical_columns:
        dummy_columns = [c for c in data_dummified.columns if c.startswith(col + '_')]
        if len(dummy_columns) >= ref_cat_col:
            category_to_drop = dummy_columns[ref_cat_col - 1]
            if category_to_drop in data_dummified.columns:
                data_dummified.drop(columns=[category_to_drop], inplace=True)
                print(f"Dropped the {ref_cat_col}th reference category: {category_to_drop}")
            else: 
                print(f"Reference category {ref_cat_col} not found in the data.")
        else:
            print(f"Reference category {ref_cat_col} not found in the data.") 
            # data_dummified.drop(columns=[category_to_drop], inplace=True)
    print(f"Data shape after dropping the {ref_cat_col}th reference category: ", data_dummified.shape)
    return data_dummified
