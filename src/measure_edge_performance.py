import pandas as pd
import numpy as np
import torch
import time
import os
import pickle
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

project_root = Path(__file__).resolve().parent.parent

data_path = project_root / "data" / "processed" / "ml_dataset.csv"

teacher_path = project_root / "models" / "teacher_model.pth"
student_path = project_root / "models" / "student_model.pth"
distilled_path = project_root / "models" / "distilled_student_model.pth"

teacher_scaler_path = project_root / "models" / "teacher_scaler.pkl"
student_scaler_path = project_root / "models" / "student_scaler.pkl"
distilled_scaler_path = project_root / "models" / "distilled_student_scaler.pkl"


# ============================================================
# FEATURES
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


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class TeacherModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(17, 64),
            torch.nn.ReLU(),

            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),

            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),

            torch.nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.network(x)


class StudentModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(17, 16),
            torch.nn.ReLU(),

            torch.nn.Linear(16, 8),
            torch.nn.ReLU()
        )

        self.classifier = torch.nn.Linear(8, 2)

    def forward(self, x):
        embedding = self.encoder(x)
        output = self.classifier(embedding)
        return output

    def get_embedding(self, x):
        return self.encoder(x)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_class, model_path):

    model = model_class()

    state_dict = torch.load(
        model_path,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


# ============================================================
# FILE SIZE
# ============================================================

def get_file_size(path):

    size_bytes = os.path.getsize(path)

    size_kb = size_bytes / 1024

    return size_bytes, size_kb


# ============================================================
# PARAMETER COUNT
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


# ============================================================
# INFERENCE TIME
# ============================================================

def measure_inference_time(model, X):

    # Warm-up
    with torch.no_grad():

        for _ in range(10):

            _ = model(X[:100])


    # Measure multiple runs
    runs = 100

    start = time.perf_counter()

    with torch.no_grad():

        for _ in range(runs):

            _ = model(X[:100])


    end = time.perf_counter()

    total_time = end - start

    average_time = total_time / runs

    # Convert seconds to milliseconds
    average_ms = average_time * 1000

    return average_ms


# ============================================================
# MAIN
# ============================================================

print("\n==========================================")
print("EDGE PERFORMANCE ANALYSIS")
print("==========================================")

# Load dataset

df = pd.read_csv(data_path)

print("\nDataset loaded:")
print(df.shape)


# Select features

X = df[features].values.astype(np.float32)

# Use first 100 samples for latency test
X_test = torch.tensor(X[:100])

print("\nInput feature dimension:", X_test.shape[1])


# ============================================================
# LOAD MODELS
# ============================================================

print("\n==========================================")
print("LOADING MODELS")
print("==========================================")

teacher = load_model(
    TeacherModel,
    teacher_path
)

student = load_model(
    StudentModel,
    student_path
)

distilled_student = load_model(
    StudentModel,
    distilled_path
)

print("Teacher loaded successfully")
print("Student loaded successfully")
print("Distilled Student loaded successfully")


# ============================================================
# PARAMETER COUNTS
# ============================================================

teacher_params = count_parameters(teacher)

student_params = count_parameters(student)

distilled_params = count_parameters(
    distilled_student
)


# ============================================================
# FILE SIZES
# ============================================================

teacher_bytes, teacher_kb = get_file_size(
    teacher_path
)

student_bytes, student_kb = get_file_size(
    student_path
)

distilled_bytes, distilled_kb = get_file_size(
    distilled_path
)


# ============================================================
# INFERENCE TIMES
# ============================================================

print("\n==========================================")
print("MEASURING INFERENCE TIME")
print("==========================================")

teacher_time = measure_inference_time(
    teacher,
    X_test
)

student_time = measure_inference_time(
    student,
    X_test
)

distilled_time = measure_inference_time(
    distilled_student,
    X_test
)


# ============================================================
# REDUCTION CALCULATIONS
# ============================================================

parameter_reduction = (
    (teacher_params - distilled_params)
    / teacher_params
) * 100

file_size_reduction = (
    (teacher_kb - distilled_kb)
    / teacher_kb
) * 100


# ============================================================
# RESULTS
# ============================================================

print("\n==========================================")
print("MODEL SIZE COMPARISON")
print("==========================================")

print("\nTeacher:")
print("Parameters:", teacher_params)
print("File size : %.2f KB" % teacher_kb)

print("\nStudent:")
print("Parameters:", student_params)
print("File size : %.2f KB" % student_kb)

print("\nDistilled Student:")
print("Parameters:", distilled_params)
print("File size : %.2f KB" % distilled_kb)


print("\n==========================================")
print("INFERENCE TIME COMPARISON")
print("==========================================")

print(
    "Teacher inference time          : %.4f ms"
    % teacher_time
)

print(
    "Student inference time          : %.4f ms"
    % student_time
)

print(
    "Distilled Student inference time: %.4f ms"
    % distilled_time
)


print("\n==========================================")
print("EDGE OPTIMIZATION")
print("==========================================")

print(
    "Parameter reduction: %.2f %%"
    % parameter_reduction
)

print(
    "File-size reduction: %.2f %%"
    % file_size_reduction
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("FINAL EDGE MODEL SUMMARY")
print("==========================================")

print(
    "\nTeacher → Distilled Student"
)

print(
    "Parameters: %d → %d"
    % (teacher_params, distilled_params)
)

print(
    "File size: %.2f KB → %.2f KB"
    % (teacher_kb, distilled_kb)
)

print(
    "Inference time: %.4f ms → %.4f ms"
    % (teacher_time, distilled_time)
)

print(
    "\nDistilled Student produces an 8-dimensional "
    "compact representation."
)

print(
    "\nEDGE PERFORMANCE ANALYSIS COMPLETED!"
)