import sys
import time
import joblib
import torch
import numpy as np

from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# PROJECT PATH
# ============================================================

project_root = Path(__file__).resolve().parent.parent

model_path = (
    project_root
    / "models"
    / "distilled_student_model.pth"
)

scaler_path = (
    project_root
    / "models"
    / "distilled_student_scaler.pkl"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Knowledge Distillation IoT API",
    description="Lightweight semantic encoder for Edge IoT devices",
    version="1.0"
)


# ============================================================
# STUDENT MODEL
# ============================================================

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

model = StudentModel()

model.load_state_dict(
    torch.load(
        model_path,
        map_location="cpu"
    )
)

model.eval()


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(scaler_path)


# ============================================================
# INPUT DATA FORMAT
# ============================================================

class SensorData(BaseModel):

    NH3: float
    NO2: float
    CO: float

    Temp: float
    Pressure: float
    Altitude: float
    Humidity: float

    CO2: float
    TVOC: float
    O3: float

    hour: int
    day: int
    month: int
    day_of_week: int

    gas_mean: float
    environment_mean: float
    pollution_mean: float


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Knowledge Distillation IoT API is running",
        "model": "Distilled Student",
        "embedding_dimension": 8,
        "parameters": 442
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(data: SensorData):

    # --------------------------------------------------------
    # CREATE FEATURE VECTOR
    # --------------------------------------------------------

    values = [[

        data.NH3,
        data.NO2,
        data.CO,

        data.Temp,
        data.Pressure,
        data.Altitude,
        data.Humidity,

        data.CO2,
        data.TVOC,
        data.O3,

        data.hour,
        data.day,
        data.month,
        data.day_of_week,

        data.gas_mean,
        data.environment_mean,
        data.pollution_mean
    ]]


    # --------------------------------------------------------
    # SCALE INPUT
    # --------------------------------------------------------

    scaled_values = scaler.transform(values)


    # Convert to PyTorch tensor

    input_tensor = torch.tensor(
        scaled_values,
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    start_time = time.perf_counter()

    with torch.no_grad():

        output = model(
            input_tensor
        )

        embedding = model.get_embedding(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

    end_time = time.perf_counter()


    # --------------------------------------------------------
    # INFERENCE TIME
    # --------------------------------------------------------

    inference_time = (
        end_time - start_time
    ) * 1000


    # --------------------------------------------------------
    # CLASS LABEL
    # --------------------------------------------------------

    if predicted_class == 0:

        condition = "Normal"

    else:

        condition = "Polluted"


    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    normal_probability = (
        probabilities[0][0].item()
    )

    polluted_probability = (
        probabilities[0][1].item()
    )


    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    embedding_values = (
        embedding[0]
        .numpy()
        .tolist()
    )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "prediction": condition,

        "class_label": predicted_class,

        "probabilities": {

            "Normal": round(
                normal_probability,
                4
            ),

            "Polluted": round(
                polluted_probability,
                4
            )
        },

        "semantic_embedding": [

            round(value, 6)

            for value in embedding_values
        ],

        "embedding_dimension": 8,

        "inference_time_ms": round(
            inference_time,
            4
        ),

        "model": "Distilled Student",

        "parameters": 442
    }