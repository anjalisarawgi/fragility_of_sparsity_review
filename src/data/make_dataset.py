import pandas as pd
import numpy as np
import os
from ucimlrepo import fetch_ucirepo

def handle_missing_values(data):
    data = data.replace('?', np.nan)
    
    # Fill missing values in 'OtherPerCap' with its mean
    data['OtherPerCap'] = data['OtherPerCap'].astype(float)
    data['OtherPerCap'] = data['OtherPerCap'].fillna(data['OtherPerCap'].mean())
    
    # Drop columns with more than 50% missing values
    initial_shape = data.shape
    data = data.dropna(thresh=len(data) * 0.5, axis=1)
    print(f"Dropped {initial_shape[1] - data.shape[1]} columns with more than 50% missing values.")
    
    return data

def summarize_data(data):
    print(data.isnull().sum())
    
    categorical = data.select_dtypes(include=['object']).columns
    numerical = data.select_dtypes(include=['int64', 'float64']).columns
    print(f"Categorical variables: {len(categorical)}")
    print(f"Numerical variables: {len(numerical)}")

def communities_and_crime_data(id: int = 183, directory: str = 'Data/raw/', filename: str = 'communities_and_crime.csv') -> pd.DataFrame:
    communities_and_crime = fetch_ucirepo(id=id) 
    
    X = communities_and_crime.data.features 
    y = communities_and_crime.data.targets 

    data = pd.concat([X, y], axis=1)
    os.makedirs(directory, exist_ok=True)
    data.to_csv(directory + filename, index=False)
    print(f"Data saved to {directory}{filename}")
    return data

def main(fetch_data: bool, id: int, raw_dir: str, processed_dir: str, filename: str, treatment_var: str, outcome_var: str):
    if fetch_data:
        data = communities_and_crime_data(id=id, directory=raw_dir, filename=filename)
    else:
        data = pd.read_csv(os.path.join(raw_dir, filename))
    
    data = handle_missing_values(data)
    summarize_data(data)

    x = data.drop(columns=[treatment_var, outcome_var])
    D = data[treatment_var]
    y = data[outcome_var]
    processed_data = pd.concat([x, D, y], axis=1)

    os.makedirs(processed_dir, exist_ok=True)
    processed_data.to_csv(os.path.join(processed_dir, filename), index=False)
    print(f"Data saved successfully to {os.path.join(processed_dir, filename)}")

if __name__ == '__main__':
    fetch_data = True  # Set to False if you don't want to fetch the data again
    id = 183
    raw_dir = 'Data/raw/'
    processed_dir = 'Data/processed/'
    filename = 'communities_and_crime.csv'
    treatment_var = 'population'
    outcome_var = 'ViolentCrimesPerPop'

    main(fetch_data, id, raw_dir, processed_dir, filename, treatment_var, outcome_var)