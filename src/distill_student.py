import pandas as pd
import numpy as np
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
# 2. LOAD DATA
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
# 7. TEACHER MODEL
# Must match the architecture used during training
# ============================================================

class TeacherModel(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 64),

            nn.ReLU(),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, 16),

            nn.ReLU(),

            nn.Linear(16, 2)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# 8. STUDENT MODEL
# ============================================================

class StudentModel(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(input_size, 16),

            nn.ReLU(),

            nn.Linear(16, 8),

            nn.ReLU()
        )

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


# ============================================================
# 9. CREATE MODELS
# ============================================================

teacher = TeacherModel(
    input_size=len(features)
)

student = StudentModel(
    input_size=len(features)
)


# ============================================================
# 10. LOAD TRAINED TEACHER
# ============================================================

teacher_path = (
    models_path
    / "teacher_model.pth"
)

teacher.load_state_dict(
    torch.load(
        teacher_path,
        map_location="cpu"
    )
)


# Teacher should not be trained again
teacher.eval()


print("\n==========================================")
print("TEACHER LOADED")
print("==========================================")

print(
    "Teacher model:",
    teacher_path
)


# ============================================================
# 11. CLASS WEIGHTS
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


# ============================================================
# 12. KNOWLEDGE DISTILLATION SETTINGS
# ============================================================

temperature = 3.0

alpha = 0.5


print("\n==========================================")
print("KNOWLEDGE DISTILLATION SETTINGS")
print("==========================================")

print(
    "Temperature (T):",
    temperature
)

print(
    "Alpha:",
    alpha
)


# ============================================================
# 13. LOSS FUNCTIONS
# ============================================================

hard_loss_function = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# 14. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    student.parameters(),
    lr=0.001
)


# ============================================================
# 15. TRAINING
# ============================================================

epochs = 30


print("\n==========================================")
print("KNOWLEDGE DISTILLATION TRAINING")
print("==========================================")

print("Epochs:", epochs)

print("\nTraining Student using Teacher knowledge...")


start_time = time.time()


for epoch in range(epochs):

    student.train()

    optimizer.zero_grad()


    # --------------------------------------------------------
    # Student predictions
    # --------------------------------------------------------

    student_logits = student(
        X_train_tensor
    )


    # --------------------------------------------------------
    # Teacher predictions
    # Teacher does NOT receive gradients
    # --------------------------------------------------------

    with torch.no_grad():

        teacher_logits = teacher(
            X_train_tensor
        )


    # --------------------------------------------------------
    # HARD LABEL LOSS
    # --------------------------------------------------------

    hard_loss = hard_loss_function(
        student_logits,
        y_train_tensor
    )


    # --------------------------------------------------------
    # SOFT TARGETS
    # --------------------------------------------------------

    student_soft = F.log_softmax(
        student_logits / temperature,
        dim=1
    )

    teacher_soft = F.softmax(
        teacher_logits / temperature,
        dim=1
    )


    # --------------------------------------------------------
    # DISTILLATION LOSS
    # --------------------------------------------------------

    distillation_loss = F.kl_div(
        student_soft,
        teacher_soft,
        reduction="batchmean"
    )


    # Temperature scaling
    distillation_loss = (
        distillation_loss
        * (temperature ** 2)
    )


    # --------------------------------------------------------
    # COMBINED LOSS
    # --------------------------------------------------------

    total_loss = (
        alpha * hard_loss
        +
        (1 - alpha) * distillation_loss
    )


    # --------------------------------------------------------
    # BACKPROPAGATION
    # --------------------------------------------------------

    total_loss.backward()

    optimizer.step()


    # --------------------------------------------------------
    # PRINT LOSS
    # --------------------------------------------------------

    if (epoch + 1) % 5 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Hard Loss: {hard_loss.item():.4f} "
            f"KD Loss: {distillation_loss.item():.4f} "
            f"Total Loss: {total_loss.item():.4f}"
        )


training_time = (
    time.time() - start_time
)


# ============================================================
# 16. EVALUATION
# ============================================================

student.eval()


with torch.no_grad():

    student_logits = student(
        X_test_tensor
    )

    predictions = torch.argmax(
        student_logits,
        dim=1
    ).numpy()


# ============================================================
# 17. METRICS
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
# 18. RESULTS
# ============================================================

print("\n==========================================")
print("DISTILLED STUDENT RESULTS")
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
# 19. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ============================================================
# 20. CLASSIFICATION REPORT
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
# 21. STUDENT PARAMETER COUNT
# ============================================================

parameter_count = sum(
    p.numel()
    for p in student.parameters()
)


parameter_memory_kb = (
    parameter_count * 4 / 1024
)


print("\n==========================================")
print("DISTILLED STUDENT SIZE")
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
# 22. TRAINING TIME
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
# 23. EXTRACT SEMANTIC EMBEDDING
# ============================================================

with torch.no_grad():

    embeddings = student.get_embedding(
        X_test_tensor
    )


print("\n==========================================")
print("DISTILLED SEMANTIC EMBEDDING")
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
# 24. SAVE DISTILLED STUDENT
# ============================================================

student_path = (
    models_path
    / "distilled_student_model.pth"
)


torch.save(
    student.state_dict(),
    student_path
)


print("\nDistilled Student model saved to:")

print(student_path)


# ============================================================
# 25. SAVE SCALER
# ============================================================

scaler_path = (
    models_path
    / "distilled_student_scaler.pkl"
)


import joblib

joblib.dump(
    scaler,
    scaler_path
)


print("\nScaler saved to:")

print(scaler_path)


# ============================================================
# 26. SAVE EMBEDDINGS
# ============================================================

embedding_path = (
    project_root
    / "results"
    / "distilled_student_embeddings.npy"
)

np.save(
    embedding_path,
    embeddings.numpy()
)


print("\nDistilled embeddings saved to:")

print(embedding_path)


# ============================================================
# 27. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("KNOWLEDGE DISTILLATION COMPLETED")
print("==========================================")