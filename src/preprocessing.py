import pandas as pd
from pathlib import Path

# ==========================================
# 1. Project and dataset paths
# ==========================================

project_root = Path(__file__).resolve().parent.parent

input_path = project_root / "data" / "raw" / "Indoor Air Pollution_Data(1).csv"
output_path = project_root / "data" / "processed" / "cleaned_air_quality.csv"


# ==========================================
# 2. Load dataset
# ==========================================

print("Loading dataset...")

df = pd.read_csv(input_path, low_memory=False)

print("Original shape:", df.shape)


# ==========================================
# 3. Remove completely empty columns
# ==========================================

df = df.dropna(axis=1, how="all")


# ==========================================
# 4. Convert sensor columns to numeric
# ==========================================

numeric_columns = [
    "NH3",
    "NO2",
    "CO",
    "PM2.5",
    "Temp",
    "Pressure",
    "Altitude",
    "Humidity",
    "CO2",
    "TVOC",
    "O3"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# ==========================================
# 5. Convert Date column
# ==========================================

df["Date"] = pd.to_datetime(
    df["Date"].str.replace("|", " ", regex=False),
    errors="coerce"
)


# ==========================================
# 6. Check missing values after conversion
# ==========================================

print("\nMissing values after conversion:")

print(df.isnull().sum())


# ==========================================
# 7. Fill missing numeric values
#    using median
# ==========================================

for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)


# ==========================================
# 8. Remove rows with invalid dates
# ==========================================

df = df.dropna(subset=["Date"])


# ==========================================
# 9. Remove duplicate rows
# ==========================================

df = df.drop_duplicates()


# ==========================================
# 10. Save cleaned dataset
# ==========================================

df.to_csv(output_path, index=False)


# ==========================================
# 11. Final information
# ==========================================

print("\n===================================")
print("CLEANING COMPLETED")
print("===================================")

print("Final shape:", df.shape)

print("\nFinal columns:")
print(df.columns.tolist())

print("\nFinal data types:")
print(df.dtypes)

print("\nFinal missing values:")
print(df.isnull().sum())

print("\nCleaned dataset saved to:")
print(output_path)