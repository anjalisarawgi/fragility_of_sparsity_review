import pandas as pd
import numpy as np
import os



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


def main(filepath, output_filepath, treatment_var, outcome_var):
    
    data = pd.read_csv(filepath)
    data = handle_missing_values(data)
    summarize_data(data)

    x = data.drop(columns=[treatment_var, outcome_var])
    D = data[treatment_var]
    y = data[outcome_var]
    processed_data = pd.concat([x, D, y], axis=1)

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    processed_data.to_csv(output_filepath, index=False)
    print(f"Data saved successfully to {output_filepath}")



if __name__ == '__main__':
    filepath = 'Data/raw/communities_and_crime.csv'
    output_filepath = 'Data/processed/communities_and_crime.csv'
    treatment_var = 'population'
    outcome_var = 'ViolentCrimesPerPop'

    main(filepath, output_filepath, treatment_var, outcome_var)
