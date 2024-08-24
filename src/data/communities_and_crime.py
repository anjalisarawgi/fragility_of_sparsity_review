import pandas as pd
import numpy as np
import os
from ucimlrepo import fetch_ucirepo
import us


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
    print("number of columns dropped: ", initial_shape[1] - data.shape[1])
    return data

def summarize_data(data):
    # print(data.isnull().sum())
    
    categorical = data.select_dtypes(include=['object']).columns
    numerical = data.select_dtypes(include=['int64', 'float64']).columns
    print(f"Categorical variables: {len(categorical)}")
    print(f"Numerical variables: {len(numerical)}")

def drop_unnecessary_columns(data, dataset_name):
    # print(f"Original data shape: {data.shape}")
    data = data.drop(columns=['communityname', 'fold', 'state'])
    # print(f"Data shape after dropping unnecessary columns: {data.shape}")
    return data

def convert_fips_to_state_abbr(data, state_column='State'):
    def fips_to_state_abbr(fips_code):
        try:
            return us.states.lookup(str(fips_code).zfill(2)).abbr
        except:
            return None
    
    data['state_abbr'] = data['State'].apply(fips_to_state_abbr)

    manual_fips_map = {
        '30': 'MT',  # Montana
        '31': 'NE',  # Nebraska
        '17': 'IL',  # Illinois
        '26': 'MI',  # Michigan
        '15': 'HI'   # Hawaii
    }

    data['state_abbr'] = data.apply(
        lambda row: manual_fips_map.get(row['State'], row['state_abbr']),
        axis=1
    )

    data['state_abbr'] = data['state_abbr'].astype('category')
    
    return data

def check_columns_with_missing_values(data):
    columns_with_missing_values = data.columns[data.isnull().any()]
    print(f"Columns with missing values: {len(columns_with_missing_values)}")
    print("number of columns with missing values: ", len(columns_with_missing_values))
    print("number of missiong values in columns: ", data[columns_with_missing_values].isnull().sum())
    print(columns_with_missing_values)


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

def main(fetch_data: bool, id: int, raw_dir: str, processed_dir: str, filename: str, treatment_var: str, outcome_var: str, dataset_name:str):
    if fetch_data:
        data = communities_and_crime_data(id=id, directory=raw_dir, filename=filename)
    else:
        data = pd.read_csv(os.path.join(raw_dir, filename))
    print("number of categorical variables: ", len(data.select_dtypes(include=['object']).columns))
    print("name of categorical variables: ", data.select_dtypes(include=['object']).columns)
    data = handle_missing_values(data)
    summarize_data(data)

    print("unique values in state column: ", data['State'].unique())
    print("length of unique values in state column: ", len(data['State'].unique()))
    print("missing values in state column: ", data['State'].isnull().sum())

    find_missing_values_by_group(data, 'State')
    print("find missing values by group: ", find_missing_values_by_group(data, 'State'))

    # drop all rows with state as MN, MI, IL, 
    data = data[~data['State'].isin(['MN', 'MI', 'IL', 'AL', 'NY', 'IA'])]
    print("data shape after dropping rows with state as MN, MI, IL: ", data.shape)

    # replace the missing vlaues in rapesPerPop with the mean of the column
    data['rapesPerPop'] = data['rapesPerPop'].astype(float)
    data['rapesPerPop'] = data['rapesPerPop'].fillna(data['rapesPerPop'].mean())
    print("missing values in rapesPerPop column: ", data['rapesPerPop'].isnull().sum())

    # do the same for robberiesPerPop, rapes, robberies, assaults, assaultPerPop, burglaries, burglariesPerPop, larcenies, larceniesPerPop, autoTheft, autoTheftPerPop, arsons, arsonsPerPop, violentPerPop, nonViolPerPop
    columns = ['robbbPerPop', 'rapes', 'robberies', 'assaults', 'assaultPerPop', 'burglaries', 'burglPerPop', 'larcenies', 'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'violentPerPop', 'nonViolPerPop']
    for col in columns:
        data[col] = data[col].astype(float)
        data[col] = data[col].fillna(data[col].mean())
        print(f"missing values in {col} column: ", data[col].isnull().sum())
    



    print("unique values in state column: ", data['State'].unique())
    print("length of unique values in state column: ", len(data['State'].unique()))
    print("missing values in state column: ", data['State'].isnull().sum())


    # check_columns_with_missing_values(data)
    check_columns_with_missing_values(data)

    # if dataset_name == "communities_and_crime" or dataset_name == "communities_and_crime_unorm":    
        # data = convert_fips_to_state_abbr(data)
    
    # data = drop_unnecessary_columns(data, dataset_name)
 
    x = data.drop(columns=[treatment_var, outcome_var])
    D = data[treatment_var]
    y = data[outcome_var]
    processed_data = pd.concat([x, D, y], axis=1)

    print("final data shape: ", processed_data.shape)
    # print("type of state_abbr: ", processed_data['state_abbr'].dtype)
    print("final number of categorical variables: ", processed_data.select_dtypes(include=['category']).columns)
    print("final number of numerical variables: ", len(processed_data.select_dtypes(include=['int64', 'float64']).columns))

    os.makedirs(processed_dir, exist_ok=True)
    data.to_csv(os.path.join(processed_dir, filename), index=False)
    print(f"Data saved successfully to {os.path.join(processed_dir, filename)}")

if __name__ == '__main__':
    fetch_data = True  # Set to False if you don't want to fetch the data again
    id = 211
    dataset_name= 'communities_and_crime_unorm'
    raw_dir = f'Data/{dataset_name}/raw'
    processed_dir = f'Data/{dataset_name}/processed/'
    filename = 'communities_and_crime_unorm.csv'
    treatment_var = 'pop' # 'population'
    outcome_var = 'violentPerPop' # 'ViolentCrimesPerPop'

    main(fetch_data, id, raw_dir, processed_dir, filename, treatment_var, outcome_var, dataset_name=dataset_name)