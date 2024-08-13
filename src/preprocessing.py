import pandas as pd
import os

def load_data(filepath):
    """Load the dataset from the given file path."""
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data, treatment_var, outcome_var):
    """Preprocess the data by separating treatment, outcome, and predictors."""
    x = data.drop(columns=[treatment_var, outcome_var])
    D = data[treatment_var]
    y = data[outcome_var]
    return x, D, y

def handle_missing_values(data):
    """Check and report missing values in the data."""
    missing_values = data.isnull().sum().sum()
    print("Total missing values: ", missing_values)
    return missing_values


def save_data(data, output_filepath):
    """Save the processed data to the specified file path."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    data.to_csv(output_filepath, index=False)
    print("Data saved successfully to", output_filepath)

def main(filepath, output_filepath, treatment_var, outcome_var):
    """Main function to execute the data processing pipeline."""
    # Load the data
    data = load_data(filepath)
    
    # Preprocess the data
    x, D, y = preprocess_data(data, treatment_var, outcome_var)
    print("Data shape: ", x.shape, "Treatment shape: ", D.shape, "Outcome shape: ", y.shape)
    
    # Handle missing values
    handle_missing_values(x)
    
    
    # Combine processed data with treatment and outcome
    processed_data = pd.concat([x, D, y], axis=1)
    
    # Save the processed data
    save_data(processed_data, output_filepath)

# Parameters for the script
filepath = 'Data/raw/communities_and_crime.csv'
output_filepath = 'Data/processed/communities_and_crime.csv'
treatment_var = 'PolicPerPop'
outcome_var = 'ViolentCrimesPerPop'

# Run the main function
if __name__ == '__main__':
    main(filepath, output_filepath, treatment_var, outcome_var)