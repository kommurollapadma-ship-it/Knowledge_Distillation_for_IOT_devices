import tkinter as tk
from tkinter import ttk, messagebox
import torch
import joblib
import numpy as np
import time
from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

project_root = Path(__file__).resolve().parent.parent

model_path = project_root / "models" / "distilled_student_model.pth"
scaler_path = project_root / "models" / "distilled_student_scaler.pkl"


# =========================================================
# CHECK MODEL FILES
# =========================================================

if not model_path.exists():
    raise FileNotFoundError(
        f"Model file not found:\n{model_path}"
    )

if not scaler_path.exists():
    raise FileNotFoundError(
        f"Scaler file not found:\n{scaler_path}"
    )


# =========================================================
# STUDENT MODEL
# =========================================================

class StudentModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder:
        # 17 input features -> 16 -> 8 embedding
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(17, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU()
        )

        # Classifier:
        # 8-dimensional embedding -> 2 classes
        self.classifier = torch.nn.Linear(8, 2)

    def forward(self, x):
        embedding = self.encoder(x)
        output = self.classifier(embedding)
        return output

    def get_embedding(self, x):
        return self.encoder(x)


# =========================================================
# LOAD MODEL
# =========================================================

model = StudentModel()

model.load_state_dict(
    torch.load(
        model_path,
        map_location="cpu"
    )
)

model.eval()

# Load scaler using joblib
scaler = joblib.load(scaler_path)


# =========================================================
# MAIN WINDOW
# =========================================================

root = tk.Tk()

root.title(
    "Knowledge Distillation - Edge IoT Monitoring"
)

root.geometry("1000x800")

root.minsize(750, 600)


# =========================================================
# SCROLLABLE AREA
# =========================================================

# Main canvas
canvas = tk.Canvas(
    root,
    highlightthickness=0
)

canvas.pack(
    side="left",
    fill="both",
    expand=True
)


# Vertical scrollbar
scrollbar = ttk.Scrollbar(
    root,
    orient="vertical",
    command=canvas.yview
)

scrollbar.pack(
    side="right",
    fill="y"
)


canvas.configure(
    yscrollcommand=scrollbar.set
)


# Frame inside canvas
main_frame = ttk.Frame(canvas)

canvas_window = canvas.create_window(
    (0, 0),
    window=main_frame,
    anchor="nw"
)


# =========================================================
# SCROLL FUNCTIONS
# =========================================================

def update_scroll_region(event=None):
    canvas.configure(
        scrollregion=canvas.bbox("all")
    )


def resize_canvas(event):
    canvas.itemconfig(
        canvas_window,
        width=event.width
    )


main_frame.bind(
    "<Configure>",
    update_scroll_region
)

canvas.bind(
    "<Configure>",
    resize_canvas
)


# Mouse wheel scrolling
def mouse_wheel(event):

    canvas.yview_scroll(
        int(-1 * (event.delta / 120)),
        "units"
    )


canvas.bind_all(
    "<MouseWheel>",
    mouse_wheel
)


# =========================================================
# TITLE
# =========================================================

title = tk.Label(
    main_frame,
    text="Knowledge-Distillation-Based\n"
         "Lightweight Semantic Encoder",
    font=("Arial", 20, "bold"),
    justify="center"
)

title.pack(
    pady=(20, 5)
)


subtitle = tk.Label(
    main_frame,
    text="Edge IoT Environmental Condition Prediction",
    font=("Arial", 13)
)

subtitle.pack(
    pady=(0, 20)
)


# =========================================================
# SENSOR INPUT FRAME
# =========================================================

input_frame = ttk.LabelFrame(
    main_frame,
    text="Sensor Input",
    padding=20
)

input_frame.pack(
    padx=30,
    pady=10,
    fill="x"
)


# =========================================================
# SENSOR INPUT FIELDS
# =========================================================

fields = [
    ("NH3", "10.0"),
    ("NO2", "5.0"),
    ("CO", "0.5"),
    ("Temperature", "25.0"),
    ("Pressure", "950.0"),
    ("Altitude", "100.0"),
    ("Humidity", "50.0"),
    ("CO2", "500.0"),
    ("TVOC", "100.0"),
    ("O3", "20.0")
]

entries = {}


for index, (name, default) in enumerate(fields):

    row = index // 2
    column = (index % 2) * 2

    label = ttk.Label(
        input_frame,
        text=name + ":",
        font=("Arial", 10)
    )

    label.grid(
        row=row,
        column=column,
        padx=15,
        pady=10,
        sticky="e"
    )

    entry = ttk.Entry(
        input_frame,
        width=25
    )

    entry.insert(
        0,
        default
    )

    entry.grid(
        row=row,
        column=column + 1,
        padx=15,
        pady=10,
        sticky="w"
    )

    entries[name] = entry


# =========================================================
# TIME FEATURES
# =========================================================

time_frame = ttk.LabelFrame(
    main_frame,
    text="Time Features",
    padding=15
)

time_frame.pack(
    padx=30,
    pady=10,
    fill="x"
)


time_fields = [
    ("Hour", "12"),
    ("Day", "15"),
    ("Month", "6"),
    ("Day of Week", "1")
]

time_entries = {}


for index, (name, default) in enumerate(time_fields):

    label = ttk.Label(
        time_frame,
        text=name + ":",
        font=("Arial", 10)
    )

    label.grid(
        row=0,
        column=index * 2,
        padx=8,
        pady=8
    )

    entry = ttk.Entry(
        time_frame,
        width=12
    )

    entry.insert(
        0,
        default
    )

    entry.grid(
        row=0,
        column=index * 2 + 1,
        padx=8,
        pady=8
    )

    time_entries[name] = entry


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict():

    try:

        # -------------------------------------------------
        # Read sensor values
        # -------------------------------------------------

        NH3 = float(entries["NH3"].get())
        NO2 = float(entries["NO2"].get())
        CO = float(entries["CO"].get())
        Temp = float(entries["Temperature"].get())
        Pressure = float(entries["Pressure"].get())
        Altitude = float(entries["Altitude"].get())
        Humidity = float(entries["Humidity"].get())
        CO2 = float(entries["CO2"].get())
        TVOC = float(entries["TVOC"].get())
        O3 = float(entries["O3"].get())

        # -------------------------------------------------
        # Read time features
        # -------------------------------------------------

        hour = int(
            time_entries["Hour"].get()
        )

        day = int(
            time_entries["Day"].get()
        )

        month = int(
            time_entries["Month"].get()
        )

        day_of_week = int(
            time_entries["Day of Week"].get()
        )

        # -------------------------------------------------
        # Derived features
        # -------------------------------------------------

        # Same feature engineering used during training
        gas_mean = np.mean([
            NH3,
            NO2,
            CO
        ])

        environment_mean = np.mean([
            Temp,
            Pressure,
            Humidity
        ])

        # PM2.5 is intentionally NOT included
        # because it is the target source.
        pollution_mean = np.mean([
            NH3,
            NO2,
            CO,
            CO2,
            TVOC,
            O3
        ])

        # -------------------------------------------------
        # 17 FEATURES
        # -------------------------------------------------

        values = [[
            NH3,
            NO2,
            CO,
            Temp,
            Pressure,
            Altitude,
            Humidity,
            CO2,
            TVOC,
            O3,
            hour,
            day,
            month,
            day_of_week,
            gas_mean,
            environment_mean,
            pollution_mean
        ]]

        # -------------------------------------------------
        # SCALE INPUT
        # -------------------------------------------------

        scaled_values = scaler.transform(
            values
        )

        input_tensor = torch.tensor(
            scaled_values,
            dtype=torch.float32
        )

        # -------------------------------------------------
        # MODEL INFERENCE
        # -------------------------------------------------

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

        inference_time = (
            end_time - start_time
        ) * 1000

        # -------------------------------------------------
        # CLASSIFICATION
        # -------------------------------------------------

        if predicted_class == 0:

            condition = "NORMAL"

        else:

            condition = "POLLUTED"

        normal_probability = (
            probabilities[0][0].item() * 100
        )

        polluted_probability = (
            probabilities[0][1].item() * 100
        )

        # -------------------------------------------------
        # GET 8-D EMBEDDING
        # -------------------------------------------------

        embedding_values = (
            embedding[0]
            .numpy()
            .tolist()
        )

        # =================================================
        # UPDATE OUTPUT
        # =================================================

        prediction_label.config(
            text=condition
        )

        probability_label.config(
            text=(
                f"Normal Probability : "
                f"{normal_probability:.2f}%\n"
                f"Polluted Probability : "
                f"{polluted_probability:.2f}%"
            )
        )

        # -------------------------------------------------
        # Embedding
        # -------------------------------------------------

        embedding_text.delete(
            "1.0",
            tk.END
        )

        embedding_text.insert(
            tk.END,
            "[\n"
            + ", ".join(
                f"{value:.6f}"
                for value in embedding_values
            )
            + "\n]"
        )

        # -------------------------------------------------
        # Model information
        # -------------------------------------------------

        info_label.config(
            text=(
                "Model: Distilled Student\n"
                "Parameters: 442\n"
                "Embedding Dimension: 8\n"
                f"Model Inference Time: "
                f"{inference_time:.4f} ms"
            )
        )

        # -------------------------------------------------
        # Status
        # -------------------------------------------------

        status_label.config(
            text="Prediction completed successfully."
        )

    except ValueError:

        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numeric values."
        )

    except Exception as error:

        messagebox.showerror(
            "Prediction Error",
            str(error)
        )


# =========================================================
# PREDICT BUTTON
# =========================================================

predict_button = ttk.Button(
    main_frame,
    text="PREDICT ENVIRONMENT CONDITION",
    command=predict
)

predict_button.pack(
    pady=20,
    ipadx=20,
    ipady=8
)


# =========================================================
# PREDICTION RESULT
# =========================================================

output_frame = ttk.LabelFrame(
    main_frame,
    text="Prediction Result",
    padding=25
)

output_frame.pack(
    padx=30,
    pady=10,
    fill="x"
)


prediction_label = tk.Label(
    output_frame,
    text="---",
    font=("Arial", 28, "bold")
)

prediction_label.pack(
    pady=10
)


probability_label = tk.Label(
    output_frame,
    text=(
        "Normal Probability : --\n"
        "Polluted Probability : --"
    ),
    font=("Arial", 14)
)

probability_label.pack(
    pady=10
)


# =========================================================
# SEMANTIC EMBEDDING
# =========================================================

embedding_frame = ttk.LabelFrame(
    main_frame,
    text="8-D Semantic Embedding",
    padding=15
)

embedding_frame.pack(
    padx=30,
    pady=10,
    fill="x"
)


embedding_text = tk.Text(
    embedding_frame,
    height=4,
    width=100,
    font=("Consolas", 11),
    wrap="word"
)

embedding_text.pack(
    fill="x"
)


# =========================================================
# MODEL INFORMATION
# =========================================================

model_frame = ttk.LabelFrame(
    main_frame,
    text="Model Information",
    padding=20
)

model_frame.pack(
    padx=30,
    pady=10,
    fill="x"
)


info_label = tk.Label(
    model_frame,
    text=(
        "Model: Distilled Student\n"
        "Parameters: 442\n"
        "Embedding Dimension: 8\n"
        "Model Inference Time: --"
    ),
    font=("Arial", 12),
    justify="center"
)

info_label.pack(
    pady=10
)


# =========================================================
# PROJECT SUMMARY
# =========================================================

summary_frame = ttk.LabelFrame(
    main_frame,
    text="Project Summary",
    padding=20
)

summary_frame.pack(
    padx=30,
    pady=10,
    fill="x"
)


summary_text = tk.Label(
    summary_frame,
    text=(
        "Knowledge Distillation-Based Lightweight Semantic Encoder\n\n"
        "Teacher Model: 3,794 parameters\n"
        "Student Model: 442 parameters\n"
        "Parameter Reduction: 88.35%\n"
        "Semantic Embedding: 8 dimensions\n"
        "Designed for Edge IoT Devices"
    ),
    font=("Arial", 11),
    justify="center"
)

summary_text.pack(
    pady=10
)


# =========================================================
# STATUS
# =========================================================

status_label = tk.Label(
    main_frame,
    text="Enter sensor values and click Predict.",
    font=("Arial", 10)
)

status_label.pack(
    pady=15
)


# =========================================================
# FOOTER
# =========================================================

footer = tk.Label(
    main_frame,
    text="Knowledge Distillation IoT Project",
    font=("Arial", 9)
)

footer.pack(
    pady=(5, 25)
)


# =========================================================
# START APPLICATION
# =========================================================

root.update_idletasks()

root.mainloop()