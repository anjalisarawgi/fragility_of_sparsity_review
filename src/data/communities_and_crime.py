import pandas as pd
import numpy as np
import os
from ucimlrepo import fetch_ucirepo
import us
from statsmodels.stats.outliers_influence import variance_inflation_factor

def create_directory(directory: str):
    os.makedirs(directory, exist_ok=True)
    print(f"Directory {directory} created or already exists.")


def handle_missing_values(data):
    data = data.replace('?', np.nan)
    
    # Fill missing values in 'OtherPerCap' with its mean
    data['otherPerCap'] = data['otherPerCap'].astype(float)
    data['otherPerCap'] = data['otherPerCap'].fillna(data['otherPerCap'].mean())
    
    # Drop columns with more than 50% missing values
    initial_shape = data.shape
    data = data.dropna(thresh=len(data) * 0.5, axis=1)
    print(f"Dropped {initial_shape[1] - data.shape[1]} columns with more than 50% missing values.")

    return data
 
def summarize_data(data):
    categorical = data.select_dtypes(include=['object']).columns
    numerical = data.select_dtypes(include=['int64', 'float64']).columns
    print(f"Categorical variables: {len(categorical)}")
    print(f"Numerical variables: {len(numerical)}")

def drop_unnecessary_columns(data, dataset_name):
    data = data.drop(columns=['communityname', 'fold', 'state'])
    return data

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

        # Handling outliers: Option 2 - Cap the outliers
        data.loc[data[col] < lower_bound, col] = lower_bound
        data.loc[data[col] > upper_bound, col] = upper_bound
        
    return data


def check_columns_with_missing_values(data):
    columns_with_missing_values = data.columns[data.isnull().any()]
    print(f"Columns with missing values: {columns_with_missing_values}")


def communities_and_crime_data(id: int, directory: str, filename: str ) -> pd.DataFrame:
    communities_and_crime = fetch_ucirepo(id=id) 
    
    X = communities_and_crime.data.features 
    y = communities_and_crime.data.targets 

    data = pd.concat([X, y], axis=1)
    
    create_directory(directory)
    data.to_csv(os.path.join(directory, filename), index=False)
    print(f"Data saved to {directory}{filename}")
    return data

# find the missing values by group state
def find_missing_values_by_group(data, group_col):
    missing_values = data.groupby(group_col).apply(lambda x: x.isnull().sum())
    # save the missing values to a csv file
    os.makedirs('Data/communities_and_crime/processed', exist_ok=True)
    missing_values.to_csv('missing_values_by_group.csv')
    return missing_values

def communities_and_crime_data(id: int, directory: str, filename: str ) -> pd.DataFrame:
    communities_and_crime = fetch_ucirepo(id=id) 
    
    X = communities_and_crime.data.features 
    y = communities_and_crime.data.targets 

    data = pd.concat([X, y], axis=1)
    
    create_directory(directory)
    data.to_csv(os.path.join(directory, filename), index=False)
    print(f"Data saved to {directory}{filename}")
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

def main(fetch_data: bool, id: int, raw_dir: str, processed_dir: str, filename: str, treatment_var: str, outcome_var: str, dataset_name:str):
    if fetch_data:
        data = communities_and_crime_data(id=id, directory=raw_dir, filename=filename)
    else:
        data = pd.read_csv(os.path.join(raw_dir, filename))
    print("data shape before processing: ", data.shape)


    data = handle_missing_values(data)
    summarize_data(data)

    # drop all rows with state as MN, MI, IL, 
    data = data[~data['State'].isin(['MN', 'MI', 'IL', 'AL', 'NY', 'IA'])]

    # do the same for robberiesPerPop, rapes, robberies, assaults, assaultPerPop, burglaries, burglariesPerPop, larcenies, larceniesPerPop, autoTheft, autoTheftPerPop, arsons, arsonsPerPop, violentPerPop, nonViolPerPop
    columns = ['rapesPerPop','robbbPerPop', 'rapes', 'robberies', 'assaults', 'assaultPerPop', 'burglaries', 'burglPerPop', 'larcenies', 'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'violentPerPop', 'nonViolPerPop']
    for col in columns:
        data[col] = data[col].astype(float)
        data[col] = data[col].fillna(data[col].mean())
 
    os.makedirs(processed_dir, exist_ok=True)
    data.to_csv(os.path.join(processed_dir, 'communities_and_crime.csv'), index=False)
    print(f"Data without checks saved to {os.path.join(processed_dir, 'data_without_checks.csv')}")

    # Prepare data for multicollinearity and outlier checks
    x = data.drop(columns=[treatment_var, outcome_var])
    D = data[treatment_var]
    y = data[outcome_var]

    # Detect and handle outliers
    data_cleaned = detect_and_handle_outliers(x.copy())
    
    # Drop high VIF columns
    data_cleaned, dropped_columns = drop_high_vif_columns(data_cleaned, threshold=10.0)
    print("Number of columns dropped due to high VIF:", len(dropped_columns))

    # Concatenate the cleaned data with treatment and outcome variables
    data_final = pd.concat([data_cleaned, D, y], axis=1)

    # Save data with multicollinearity and outlier checks
    data_final.to_csv(os.path.join(processed_dir, 'communities_and_crime_with_checks.csv'), index=False)
    print("data shape after processing: ", data_final.shape)
    print(f"Data with checks saved to {os.path.join(processed_dir, 'data_with_checks.csv')}")


if __name__ == '__main__':
    fetch_data = True  # Set to False if you don't want to fetch the data again
    id = 211
    dataset_name= 'communities_and_crime'
    raw_dir = f'Data/{dataset_name}/raw'
    processed_dir = f'Data/{dataset_name}/processed/'
    filename = 'communities_and_crime.csv'
    treatment_var = 'pop' # 'population'
    outcome_var = 'violentPerPop' # 'ViolentCrimesPerPop'

    main(fetch_data, id, raw_dir, processed_dir, filename, treatment_var, outcome_var, dataset_name=dataset_name)