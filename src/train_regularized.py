import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Regularized Logistic Regression
model = LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n==========================================")
print("REGULARIZED LOGISTIC REGRESSION RESULTS")
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

print("\n==========================================")
print("REGULARIZED MODEL COMPLETED")
print("==========================================")