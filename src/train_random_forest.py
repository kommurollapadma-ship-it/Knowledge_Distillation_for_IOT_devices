import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

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
y = df["Condition_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n==========================================")
print("RANDOM FOREST RESULTS")
print("==========================================")

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Polluted"],
        zero_division=0
    )
)

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\n==========================================")
print("TOP 10 FEATURE sIMPORTANCES")
print("==========================================")

print(importance.head(10))

print("\n==========================================")
print("RANDOM FOREST COMPLETED")
print("==========================================")