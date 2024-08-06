import os
import subprocess
import zipfile
import pandas as pd
import torch
import kaggle
# NOTE: Kaggle API must be installed and kaggle.json must be configured

def download_and_process_data():
    subprocess.run(["kaggle", "datasets", "download", "-d", "thedevastator/jobs-dataset-from-glassdoor"])

    zip_file = "jobs-dataset-from-glassdoor.zip"
    extract_path = "data/raw/"
    processed_path = "data/processed/"

    os.makedirs(extract_path, exist_ok=True)
    os.makedirs(processed_path, exist_ok=True)

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    os.remove(zip_file)

    file_path = os.path.join(extract_path, "glassdoor_jobs.csv")
    df = pd.read_csv(file_path)

    print(df.head())
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df_tensor = torch.tensor(df.values, dtype=torch.float32)
    torch.save(df_tensor, os.path.join(processed_path, "jobs_data.pt"))

    print("Data saved successfully!")

if __name__ == "__main__":
    download_and_process_data()
