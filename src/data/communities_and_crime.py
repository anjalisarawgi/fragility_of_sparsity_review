from ucimlrepo import fetch_ucirepo 
import pandas as pd
import os


def communities_and_crime_data(id: int = 183, directory: str = 'Data/raw/', filename: str = 'communities_and_crime.csv') -> pd.DataFrame:
    communities_and_crime = fetch_ucirepo(id=id) 
    
    X = communities_and_crime.data.features 
    y = communities_and_crime.data.targets 
    
    # print(communities_and_crime.metadata) 
    # print(communities_and_crime.variables) 

    data = pd.concat([X, y], axis=1)
    os.makedirs(directory, exist_ok=True)
    data.to_csv(directory + filename, index=False)
    print(f"Data saved to {directory}{filename}")
    return data

if __name__ == '__main__':
    communities_and_crime_data()