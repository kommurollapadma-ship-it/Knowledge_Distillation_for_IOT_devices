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
# 3. FEATURES
# PM2.5 IS EXCLUDED TO AVOID TARGET LEAKAGE
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
# 6. PYTORCH TENSORS
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
# 7. LIGHTWEIGHT STUDENT MODEL
# ============================================================

class StudentModel(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        # Encoder creates the compact representation
        self.encoder = nn.Sequential(

            nn.Linear(input_size, 16),

            nn.ReLU(),

            nn.Linear(16, 8),

            nn.ReLU()
        )

        # Classification head
        self.classifier = nn.Linear(
            8,
            2
        )


    def forward(self, x):

        embedding = self.encoder(x)

        output = self.classifier(
            embedding
        )

        return output


    def get_embedding(self, x):

        return self.encoder(x)


# Create Student

student = StudentModel(
    input_size=len(features)
)


print("\n==========================================")
print("STUDENT MODEL")
print("==========================================")

print(student)


# ============================================================
# 8. CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    y_train
)

normal_count = class_counts[0]

polluted_count = class_counts[1]

total_samples = len(y_train)


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
    student.parameters(),
    lr=0.001
)


# ============================================================
# 11. TRAINING
# ============================================================

epochs = 30


print("\n==========================================")
print("STUDENT TRAINING")
print("==========================================")

print("Epochs:", epochs)

print("Learning rate: 0.001")

print("\nTraining lightweight student...")


start_time = time.time()


for epoch in range(epochs):

    student.train()

    optimizer.zero_grad()

    # Forward pass
    outputs = student(
        X_train_tensor
    )

    # Hard-label loss
    loss = criterion(
        outputs,
        y_train_tensor
    )

    # Backpropagation
    loss.backward()

    optimizer.step()


    if (epoch + 1) % 5 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss.item():.4f}"
        )


training_time = (
    time.time() - start_time
)


# ============================================================
# 12. EVALUATION
# ============================================================

student.eval()


with torch.no_grad():

    outputs = student(
        X_test_tensor
    )

    predictions = torch.argmax(
        outputs,
        dim=1
    ).numpy()


# ============================================================
# 13. METRICS
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
# 14. RESULTS
# ============================================================

print("\n==========================================")
print("STUDENT MODEL RESULTS")
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
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 16. CLASSIFICATION REPORT
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
# 17. PARAMETER COUNT
# ============================================================

parameter_count = sum(
    p.numel()
    for p in student.parameters()
)


parameter_memory_kb = (
    parameter_count * 4 / 1024
)


print("\n==========================================")
print("STUDENT MODEL SIZE")
print("==========================================")

print(
    "Total parameters:",
    parameter_count
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
# 18. TRAINING TIME
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
# 19. TEST EMBEDDING
# ============================================================

with torch.no_grad():

    embeddings = student.get_embedding(
        X_test_tensor
    )


print("\n==========================================")
print("SEMANTIC EMBEDDING")
print("==========================================")

print(
    "Embedding shape:",
    embeddings.shape
)

print(
    "Embedding dimension:",
    embeddings.shape[1]
)


# ============================================================
# 20. SAVE STUDENT MODEL
# ============================================================

student_path = (
    models_path
    / "student_model.pth"
)


torch.save(
    student.state_dict(),
    student_path
)


print("\nStudent model saved to:")

print(student_path)


# ============================================================
# 21. SAVE STUDENT SCALER
# ============================================================

scaler_path = (
    models_path
    / "student_scaler.pkl"
)


joblib.dump(
    scaler,
    scaler_path
)


print("\nScaler saved to:")

print(scaler_path)


# ============================================================
# 22. SAVE EMBEDDINGS
# ============================================================

embedding_array = embeddings.numpy()

embedding_path = (
    project_root
    / "results"
    / "student_embeddings.npy"
)

embedding_path.parent.mkdir(
    exist_ok=True
)

np.save(
    embedding_path,
    embedding_array
)


print("\nEmbeddings saved to:")

print(embedding_path)


# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("STUDENT TRAINING COMPLETED")
print("==========================================")