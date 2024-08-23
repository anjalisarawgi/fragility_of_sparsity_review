from causalinference import CausalModel
import pandas as pd
import dowhy.datasets
import os


def download_lalonde(directory ='Data/lalonde/raw', filename='raw_lalonde.csv'):
    lalonde = dowhy.datasets.lalonde_dataset()
    os.makedirs(directory, exist_ok=True)
    lalonde.to_csv(os.path.join(directory, filename), index=False)
    print(f"Data saved to {os.path.join(directory, filename)}")
    return lalonde

# def processed_lalonde(data):
#     data["treat"] = data["treat"].astype(int)

#     for column in ['black', 'hisp', 'married', 'nodegr']:
#         data[column] = data[column].astype(int)
#     data_encoded = pd.get_dummies(data, columns=['black', 'hisp', 'married', 'nodegr'], drop_first=False)
    
#     # Ensuring binary variables true and false are integers 0 and 1 
#     binary_cols = [col for col in data_encoded.columns if col.endswith('_1') or col.endswith('_0')]
#     data_encoded[binary_cols] = data_encoded[binary_cols].astype(int)
#         # Convert specific columns to categorical variables
#     categorical_columns = ['treat'] + binary_cols
#     for col in categorical_columns:
#         data_encoded[col] = data_encoded[col].astype('category')
    
#     print(f"Data shape after processing: {data_encoded.shape}")
#     print("Categorical columns:", categorical_columns)
    

#     # convert to categorical
    
    
#     print(f"Data shape after processing: {data_encoded.shape}")
    
#     return data_encoded

def processed_lalonde(data):
    columns_to_convert = ['black', 'hisp', 'married', 'nodegr']
    data[columns_to_convert] = data[columns_to_convert].astype('category') 
    return data



def save_processed_data(data, directory='Data/lalonde/processed', filename='lalonde.csv'):
    os.makedirs(directory, exist_ok=True)
    data.to_csv(os.path.join(directory, filename), index=False)
    print(f"Data saved to {os.path.join(directory, filename)}")
    return

def main(fetch_data = True, raw_dir = 'Data/lalonde/raw', processed_dir = 'Data/lalonde/processed'):
    if fetch_data:
        data = download_lalonde(directory=raw_dir)
    else:
        data = pd.read_csv(os.path.join(raw_dir, 'raw_lalonde.csv'))
    
    data_processed = processed_lalonde(data)
    print("data ---- types:", data.dtypes)
    save_processed_data(data_processed, directory=processed_dir)
    
    return data_processed

if  __name__ == '__main__':
    main()
    data = pd.read_csv('Data/lalonde/processed/lalonde.csv')
    print(data.head())