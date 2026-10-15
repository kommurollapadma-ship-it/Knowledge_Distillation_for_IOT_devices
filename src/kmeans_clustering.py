import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Project root
project_root = Path(__file__).resolve().parent.parent

# Input dataset
input_path = project_root / "data" / "processed" / "ml_dataset.csv"


# Load dataset
df = pd.read_csv(input_path)

print("Dataset shape:", df.shape)


# Same features used in our ML models
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


# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print("\nFeatures scaled successfully.")


# -------------------------------------------------
# K-MEANS CLUSTERING
# -------------------------------------------------

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)


# Add cluster result
df["Cluster"] = clusters


print("\n==========================================")
print("K-MEANS CLUSTERING RESULTS")
print("==========================================")


print("\nCluster counts:")
print(df["Cluster"].value_counts().sort_index())


# -------------------------------------------------
# SILHOUETTE SCORE
# -------------------------------------------------

# Calculate silhouette score on a sample
sample_size = min(10000, len(X_scaled))

sample = pd.DataFrame(X_scaled).sample(
    n=sample_size,
    random_state=42
)

sample_labels = kmeans.predict(sample)

silhouette = silhouette_score(
    sample,
    sample_labels
)

print("\nSilhouette Score:", round(silhouette, 4))


# -------------------------------------------------
# COMPARE CLUSTERS WITH OUR TARGET
# -------------------------------------------------

print("\nCluster vs Condition:")
print(
    pd.crosstab(
        df["Cluster"],
        df["Condition"]
    )
)


# -------------------------------------------------
# SAVE RESULT
# -------------------------------------------------

output_path = project_root / "data" / "processed" / "kmeans_results.csv"

df.to_csv(output_path, index=False)


print("\nResults saved to:")
print(output_path)