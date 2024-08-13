import pandas as pd
import numpy as np
import os

def load_data(filepath):
    """Load the dataset from the given file path."""
    try:
        data = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def preprocess_data(data, treatment_var, outcome_var):
    """Preprocess the data by separating treatment, outcome, and predictors."""
    x = data.drop(columns=[treatment_var, outcome_var])
    D = data[treatment_var]
    y = data[outcome_var]
    return x, D, y

def handle_missing_values(data):
    """Handle missing values in the dataset."""
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
    """Print summary information about the data."""
    print("Summary of missing values per column:")
    print(data.isnull().sum())
    print("\nSummary of categorical and numerical variables:")
    categorical = data.select_dtypes(include=['object']).columns
    numerical = data.select_dtypes(include=['int64', 'float64']).columns
    print(f"Categorical variables: {len(categorical)}")
    print(f"Numerical variables: {len(numerical)}")

def save_data(data, output_filepath):
    """Save the processed data to the specified file path."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    data.to_csv(output_filepath, index=False)
    print(f"Data saved successfully to {output_filepath}")

def main(filepath, output_filepath, treatment_var, outcome_var):
    """Main function to execute the data processing pipeline."""
    # Load the data
    data = load_data(filepath)
    if data is None:
        return

    # Handle missing values
    data = handle_missing_values(data)

    # Summarize data
    summarize_data(data)

    # Preprocess the data
    x, D, y = preprocess_data(data, treatment_var, outcome_var)
    
    # Combine processed data with treatment and outcome
    processed_data = pd.concat([x, D, y], axis=1)
    
    # Save the processed data
    save_data(processed_data, output_filepath)
    print("Data processing pipeline completed successfully.")

# Parameters for the script
filepath = 'Data/raw/communities_and_crime.csv'
output_filepath = 'Data/processed/communities_and_crime.csv'
treatment_var = 'population'
outcome_var = 'ViolentCrimesPerPop'

# Run the main function
if __name__ == '__main__':
    main(filepath, output_filepath, treatment_var, outcome_var)
