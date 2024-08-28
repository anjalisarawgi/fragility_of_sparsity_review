import pandas as pd
data = pd.read_csv('Data/lalonde/processed/lalonde.csv')
print("number of rows: ", data.shape[0])
print("number of columns: ", data.shape[1])