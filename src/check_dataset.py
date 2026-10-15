import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

dataset_path = project_root / "data" / "raw" / "Indoor Air Pollution_Data(1).csv"

print("Dataset path:")
print(dataset_path)

df = pd.read_csv(dataset_path)

print("\n==============================")
print("DATASET INFORMATION")
print("==============================")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe())