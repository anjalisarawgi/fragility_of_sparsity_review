import openml
import pandas as pd
import os

dataset = openml.datasets.get_dataset(42713)
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
os.makedirs('Data/bikeshare', exist_ok=True)
df_features.to_csv('Data/bikeshare/bikeshare.csv', index=False)


print("CSV files have been saved to the 'Data/bikeshare/' directory.")