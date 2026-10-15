import pandas as pd
from pathlib import Path

# Project folder
project_root = Path(__file__).resolve().parent.parent

# Cleaned dataset
dataset_path = (
    project_root
    / "data"
    / "processed"
    / "cleaned_air_quality.csv"
)

# Load dataset
df = pd.read_csv(dataset_path)

print("========================================")
print("EXPLORATORY DATA ANALYSIS")
print("========================================")

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nStatistical summary:")
print(df.describe().T)

print("\nMinimum values:")
print(df.min(numeric_only=True))

print("\nMaximum values:")
print(df.max(numeric_only=True))

print("\nMedian values:")
print(df.median(numeric_only=True))

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of unique values:")
print(df.nunique())

print("\n========================================")
print("CHECKING POSSIBLE SENSOR ANOMALIES")
print("========================================")

# Check negative values
numeric_df = df.select_dtypes(include="number")

print("\nNegative value counts:")
print((numeric_df < 0).sum())

print("\n========================================")
print("CORRELATION WITH PM2.5")
print("========================================")

if "PM2.5" in df.columns:
    correlation = df[numeric_df.columns].corr()["PM2.5"].sort_values(
        ascending=False
    )
    print(correlation)

print("\nEDA completed successfully.")