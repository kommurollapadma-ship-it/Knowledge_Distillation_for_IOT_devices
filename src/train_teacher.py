import pandas as pd
import numpy as np
import time
import joblib
from pathlib import Path

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "data"
    / "processed"
    / "ml_dataset.csv"
)

models_path = project_root / "models"

models_path.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(input_path)

print("==========================================")
print("LOADING DATASET")
print("==========================================")

print("Dataset shape:", df.shape)


# ============================================================
# 3. SELECT FEATURES
# IMPORTANT:
# PM2.5 is NOT included because PM2.5 was used
# to create our Condition target.
# ============================================================

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


X = df[features].values

y = df["Condition_Label"].values


print("\nNumber of input features:", len(features))

print("\nTarget distribution:")

print(
    pd.Series(y).value_counts().sort_index()
)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n==========================================")
print("DATA SPLIT")
print("==========================================")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 5. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


print("\nFeature scaling completed.")


# ============================================================
# 6. CONVERT DATA TO PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)


# ============================================================
# 7. TEACHER MODEL
# ============================================================

class TeacherModel(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            # Input layer
            nn.Linear(input_size, 64),

            nn.ReLU(),

            # Hidden layer
            nn.Linear(64, 32),

            nn.ReLU(),

            # Embedding layer
            nn.Linear(32, 16),

            nn.ReLU(),

            # Output layer
            nn.Linear(16, 2)
        )

    def forward(self, x):

        return self.network(x)


# Create teacher model

teacher = TeacherModel(
    input_size=len(features)
)


print("\n==========================================")
print("TEACHER MODEL")
print("==========================================")

print(teacher)


# ============================================================
# 8. HANDLE CLASS IMBALANCE
# ============================================================

class_counts = np.bincount(y_train)

normal_count = class_counts[0]

polluted_count = class_counts[1]

total_samples = len(y_train)


# Balanced class weights

normal_weight = (
    total_samples /
    (2 * normal_count)
)

polluted_weight = (
    total_samples /
    (2 * polluted_count)
)


class_weights = torch.tensor(
    [
        normal_weight,
        polluted_weight
    ],
    dtype=torch.float32
)


print("\n==========================================")
print("CLASS WEIGHTS")
print("==========================================")

print(
    "Normal weight   :",
    round(normal_weight, 4)
)

print(
    "Polluted weight :",
    round(polluted_weight, 4)
)


# ============================================================
# 9. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# 10. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    teacher.parameters(),
    lr=0.001
)


# ============================================================
# 11. TRAINING SETTINGS
# ============================================================

epochs = 30


print("\n==========================================")
print("TEACHER TRAINING")
print("==========================================")

print("Epochs:", epochs)

print("Learning rate: 0.001")

print("\nTraining teacher model...")


# ============================================================
# 12. TRAIN TEACHER MODEL
# ============================================================

start_time = time.time()


for epoch in range(epochs):

    teacher.train()

    # Clear old gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = teacher(
        X_train_tensor
    )

    # Calculate loss
    loss = criterion(
        outputs,
        y_train_tensor
    )

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()


    # Print every 5 epochs

    if (epoch + 1) % 5 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss.item():.4f}"
        )


training_time = (
    time.time() - start_time
)


# ============================================================
# 13. EVALUATE TEACHER
# ============================================================

teacher.eval()


with torch.no_grad():

    outputs = teacher(
        X_test_tensor
    )

    predictions = torch.argmax(
        outputs,
        dim=1
    ).numpy()


# ============================================================
# 14. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


# ============================================================
# 15. DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("TEACHER MODEL RESULTS")
print("==========================================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Normal",
            "Polluted"
        ],
        zero_division=0
    )
)


# ============================================================
# 18. MODEL PARAMETER COUNT
# ============================================================

parameter_count = sum(
    p.numel()
    for p in teacher.parameters()
)


print("\n==========================================")
print("MODEL SIZE INFORMATION")
print("==========================================")

print(
    "Total parameters:",
    parameter_count
)


# Each float32 parameter ≈ 4 bytes

parameter_memory_kb = (
    parameter_count * 4 / 1024
)


print(
    "Approximate parameter memory:",
    round(
        parameter_memory_kb,
        2
    ),
    "KB"
)


# ============================================================
# 19. TRAINING TIME
# ============================================================

print(
    "\nTraining time:",
    round(
        training_time,
        2
    ),
    "seconds"
)


# ============================================================
# 20. SAVE TEACHER MODEL
# ============================================================

teacher_path = (
    models_path
    / "teacher_model.pth"
)


torch.save(
    teacher.state_dict(),
    teacher_path
)


print("\nTeacher model saved to:")

print(teacher_path)


# ============================================================
# 21. SAVE SCALER
# ============================================================

scaler_path = (
    models_path
    / "teacher_scaler.pkl"
)


joblib.dump(
    scaler,
    scaler_path
)


print("\nScaler saved to:")

print(scaler_path)


# ============================================================
# 22. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("TEACHER TRAINING COMPLETED")
print("==========================================")