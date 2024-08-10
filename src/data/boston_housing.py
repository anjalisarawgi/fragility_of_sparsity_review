import openml
import pandas as pd
import os

dataset = openml.datasets.get_dataset(531)
df, *_ = dataset.get_data()

print(df.head())
print(df)
print("description of the data", df.describe())
if 'label' in df.columns:
    df_features = df.drop(columns=['label'])
    df_labels = df['label']
else:
    df_features = df
    df_labels = pd.Series()  # Empty Series if no labels

# Save to CSV files
os.makedirs('Data/boston_housing', exist_ok=True)
df_features.to_csv('Data/boston_housing/boston_housing.csv', index=False)

print("CSV files have been saved to the 'Data/boston_housing/' directory.")