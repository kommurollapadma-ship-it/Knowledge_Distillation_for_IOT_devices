import pandas as pd
from pathlib import Path

# ======================================
# 1. Project and dataset path
# ======================================

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "data"
    / "processed"
    / "cleaned_air_quality.csv"
)


# ======================================
# 2. Load dataset
# ======================================

df = pd.read_csv(input_path)


# ======================================
# 3. PM2.5 Analysis
# ======================================

print("======================================")
print("PM2.5 ANALYSIS")
print("======================================")


# Number of unique values
print("\nNumber of unique PM2.5 values:")
print(df["PM2.5"].nunique())


# Most common values
print("\nMost common PM2.5 values:")
print(df["PM2.5"].value_counts().head(20))


# Percentage of zero values
print("\nPercentage of zero PM2.5:")

zero_percentage = (
    (df["PM2.5"] == 0).mean() * 100
)

print(f"{zero_percentage:.2f}%")


# Quantiles
print("\nPM2.5 quantiles:")

print(
    df["PM2.5"].quantile(
        [
            0,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.00
        ]
    )
)


# Descriptive statistics
print("\nPM2.5 descriptive statistics:")

print(df["PM2.5"].describe())


# ======================================
# 4. Check missing values
# ======================================

print("\nMissing PM2.5 values:")

print(df["PM2.5"].isnull().sum())


# ======================================
# 5. Check negative values
# ======================================

print("\nNegative PM2.5 values:")

print((df["PM2.5"] < 0).sum())


# ======================================
# 6. Final message
# ======================================

print("\n======================================")
print("PM2.5 ANALYSIS COMPLETED")
print("======================================")