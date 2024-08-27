# from causalinference import CausalModel
# import pandas as pd
# import dowhy.datasets
# import os


# def download_lalonde(directory ='Data/lalonde/raw', filename='raw_lalonde.csv'):
#     lalonde = dowhy.datasets.lalonde_dataset()
#     os.makedirs(directory, exist_ok=True)
#     lalonde.to_csv(os.path.join(directory, filename), index=False)
#     print(f"Data saved to {os.path.join(directory, filename)}")
#     return lalonde

# # def processed_lalonde(data):
# #     data["treat"] = data["treat"].astype(int)

# #     for column in ['black', 'hisp', 'married', 'nodegr']:
# #         data[column] = data[column].astype(int)
# #     data_encoded = pd.get_dummies(data, columns=['black', 'hisp', 'married', 'nodegr'], drop_first=False)
    
# #     # Ensuring binary variables true and false are integers 0 and 1 
# #     binary_cols = [col for col in data_encoded.columns if col.endswith('_1') or col.endswith('_0')]
# #     data_encoded[binary_cols] = data_encoded[binary_cols].astype(int)
# #         # Convert specific columns to categorical variables
# #     categorical_columns = ['treat'] + binary_cols
# #     for col in categorical_columns:
# #         data_encoded[col] = data_encoded[col].astype('category')
    
# #     print(f"Data shape after processing: {data_encoded.shape}")
# #     print("Categorical columns:", categorical_columns)
    

# #     # convert to categorical
    
    
# #     print(f"Data shape after processing: {data_encoded.shape}")
    
# #     return data_encoded

# def processed_lalonde(data):
#     columns_to_convert = ['black', 'hisp', 'married', 'nodegr']
#     data[columns_to_convert] = data[columns_to_convert].astype('category') 
#     return data



# def save_processed_data(data, directory='Data/lalonde/processed', filename='lalonde.csv'):
#     os.makedirs(directory, exist_ok=True)
#     data.to_csv(os.path.join(directory, filename), index=False)
#     print(f"Data saved to {os.path.join(directory, filename)}")
#     return

# def main(fetch_data = True, raw_dir = 'Data/lalonde/raw', processed_dir = 'Data/lalonde/processed'):
#     if fetch_data:
#         data = download_lalonde(directory=raw_dir)
#     else:
#         data = pd.read_csv(os.path.join(raw_dir, 'raw_lalonde.csv'))
    
#     data_processed = processed_lalonde(data)
#     print("data ---- types:", data.dtypes)
#     save_processed_data(data_processed, directory=processed_dir)
    
#     return data_processed

# if  __name__ == '__main__':
#     main()
#     data = pd.read_csv('Data/lalonde/processed/lalonde.csv')
#     print(data.head())

from causalinference import CausalModel
import pandas as pd
import dowhy.datasets
import os
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

def download_lalonde(directory='Data/lalonde/raw', filename='raw_lalonde.csv'):
    lalonde = dowhy.datasets.lalonde_dataset()
    os.makedirs(directory, exist_ok=True)
    lalonde.to_csv(os.path.join(directory, filename), index=False)
    print(f"Data saved to {os.path.join(directory, filename)}")
    return lalonde

def processed_lalonde(data):
    columns_to_convert = ['black', 'hisp', 'married', 'nodegr']
    data[columns_to_convert] = data[columns_to_convert].astype('category')
    return data

def save_processed_data(data, directory='Data/lalonde/processed', filename='lalonde.csv'):
    os.makedirs(directory, exist_ok=True)
    data.to_csv(os.path.join(directory, filename), index=False)
    print(f"Data saved to {os.path.join(directory, filename)}")
    return

# Function to detect and handle outliers using the IQR method
def detect_and_handle_outliers(data):
    for col in data.select_dtypes(include=[np.number]).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Detect outliers
        outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
        num_outliers = outliers.sum()
        print(f"Column {col} has {num_outliers} outliers.")

        # Handling outliers: Option 1 - Remove outliers
        # data = data[~outliers]

        # Handling outliers: Option 2 - Cap the outliers
        data.loc[data[col] < lower_bound, col] = lower_bound
        data.loc[data[col] > upper_bound, col] = upper_bound
        
    return data

# Function to calculate VIF and drop high VIF columns
def calculate_vif(data: pd.DataFrame):
    data = data.select_dtypes(include=[np.number])
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(len(data.columns))]
    return vif_data

def drop_high_vif_columns(data: pd.DataFrame, threshold: float = 10.0):
    high_vif_columns = []
    while True:
        vif_data = calculate_vif(data)
        high_vif = vif_data[vif_data["VIF"] > threshold]
        if high_vif.empty:
            break
        else:
            col_to_drop = high_vif.sort_values("VIF", ascending=False).iloc[0]["feature"]
            print(f"Dropping column {col_to_drop} with VIF {high_vif['VIF'].max()}")
            data = data.drop(columns=[col_to_drop])
            high_vif_columns.append(col_to_drop)
    return data, high_vif_columns

def main(fetch_data=True, raw_dir='Data/lalonde/raw', processed_dir='Data/lalonde/processed'):
    if fetch_data:
        data = download_lalonde(directory=raw_dir)
    else:
        data = pd.read_csv(os.path.join(raw_dir, 'raw_lalonde.csv'))
    
    # Process the dataset without cleaning
    data_no_cleaning = processed_lalonde(data)
    save_processed_data(data_no_cleaning, directory=processed_dir, filename='lalonde.csv')
    
    # Detect and handle outliers
    data_cleaned = detect_and_handle_outliers(data_no_cleaning.copy())
    
    # Check for multicollinearity and drop columns with VIF > 10
    data_cleaned, dropped_columns = drop_high_vif_columns(data_cleaned, threshold=10.0)
    print("Number of columns dropped due to high VIF:", len(dropped_columns))
    
    # Save the cleaned dataset
    save_processed_data(data_cleaned, directory=processed_dir, filename='lalonde_cleaned.csv')
    
    return data_no_cleaning, data_cleaned

if __name__ == '__main__':
    data_no_cleaning, data_cleaned = main()
    print("Data without cleaning:")
    print(data_no_cleaning.head())
    print("Data with cleaning:")
    print(data_cleaned.head())
