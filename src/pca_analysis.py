import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

project_root = Path(__file__).resolve().parent.parent

dataset_path = (
    project_root
    / "data"
    / "processed"
    / "ml_dataset.csv"
)

print("Loading dataset...")

df = pd.read_csv(dataset_path)

features = [
    "NH3",
    "NO2",
    "CO",
    "Temp",
    "Pressure",
    "Altitude",
    "Humidity",
    "CO2",
    "TVOC",
    "O3",
    "hour",
    "day",
    "month",
    "day_of_week",
    "gas_mean",
    "environment_mean",
    "pollution_mean"
]

X = df[features]

print("Original feature shape:", X.shape)

# Step 1: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\n==========================================")
print("PCA RESULTS")
print("==========================================")

print("\nPCA shape:", X_pca.shape)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal variance explained by 2 components:")
print(f"{pca.explained_variance_ratio_.sum() * 100:.2f}%")

print("\nPrincipal Component 1:")
print(f"{pca.explained_variance_ratio_[0] * 100:.2f}%")

print("\nPrincipal Component 2:")
print(f"{pca.explained_variance_ratio_[1] * 100:.2f}%")

print("\n==========================================")
print("PCA COMPLETED")
print("==========================================")