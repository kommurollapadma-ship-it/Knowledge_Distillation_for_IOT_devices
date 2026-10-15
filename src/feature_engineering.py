import pandas as pd
from pathlib import Path

# ==========================================
# 1. Paths
# ==========================================

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "data"
    / "processed"
    / "cleaned_air_quality.csv"
)

output_path = (
    project_root
    / "data"
    / "processed"
    / "ml_dataset.csv"
)


# ==========================================
# 2. Load cleaned dataset
# ==========================================

df = pd.read_csv(input_path)

print("Original shape:", df.shape)


# ==========================================
# 3. Convert Date
# ==========================================

df["Date"] = pd.to_datetime(df["Date"])


# ==========================================
# 4. Extract time-based features
# ==========================================

df["hour"] = df["Date"].dt.hour
df["day"] = df["Date"].dt.day
df["month"] = df["Date"].dt.month
df["day_of_week"] = df["Date"].dt.dayofweek


# ==========================================
# 5. Create sensor features
# ==========================================

# Average of gas sensors
df["gas_mean"] = df[
    ["NH3", "NO2", "CO"]
].mean(axis=1)


# Average environmental conditions
df["environment_mean"] = df[
    ["Temp", "Pressure", "Humidity"]
].mean(axis=1)


# IMPORTANT:
# Do NOT include PM2.5 here because PM2.5
# is used to create our target.
df["pollution_mean"] = df[
    ["NH3", "NO2", "CO", "CO2", "TVOC", "O3"]
].mean(axis=1)


# ==========================================
# 6. Create binary target
# ==========================================

# PM2.5 = 0  -> Normal
# PM2.5 > 0  -> Polluted

df["Condition"] = df["PM2.5"].apply(
    lambda x: "Normal" if x == 0 else "Polluted"
)


# ==========================================
# 7. Convert target to numbers
# ==========================================

condition_mapping = {
    "Normal": 0,
    "Polluted": 1
}

df["Condition_Label"] = df["Condition"].map(
    condition_mapping
)


# ==========================================
# 8. Remove original Date
# ==========================================

df = df.drop(columns=["Date"])


# ==========================================
# 9. Save ML dataset
# ==========================================

df.to_csv(output_path, index=False)


# ==========================================
# 10. Display results
# ==========================================

print("\n==========================================")
print("FEATURE ENGINEERING COMPLETED")
print("==========================================")

print("\nFinal shape:")
print(df.shape)

print("\nFinal columns:")
print(df.columns.tolist())

print("\nCondition distribution:")
print(df["Condition"].value_counts())

print("\nCondition percentages:")
print(
    df["Condition"].value_counts(normalize=True) * 100
)

print("\nTarget labels:")
print(df["Condition_Label"].value_counts().sort_index())

print("\nSample data:")
print(df.head())

print("\nSaved to:")
print(output_path)