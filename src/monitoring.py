import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

input_path = project_root / "data" / "processed" / "ml_dataset.csv"
output_path = project_root / "results" / "drift_report.csv"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(input_path)

print("==========================================")
print("ML MODEL MONITORING & DRIFT DETECTION")
print("==========================================")

print("\nDataset shape:", df.shape)

# --------------------------------------------------
# FEATURES USED BY THE MODEL
# --------------------------------------------------

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

# --------------------------------------------------
# REFERENCE AND CURRENT DATA
# --------------------------------------------------

reference_size = 5000
current_size = 5000

reference_data = df.iloc[:reference_size][features]
current_data = df.iloc[-current_size:][features]

print("\nReference samples:", len(reference_data))
print("Current samples:", len(current_data))

# --------------------------------------------------
# PSI FUNCTION
# --------------------------------------------------

def calculate_psi(reference, current, bins=10):

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    # Remove invalid values
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    # Create bins using reference data
    breakpoints = np.percentile(
        reference,
        np.linspace(0, 100, bins + 1)
    )

    breakpoints = np.unique(breakpoints)

    # If feature has very few unique values
    if len(breakpoints) < 3:
        return 0.0

    # Add infinity boundaries
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference,
        bins=breakpoints
    )

    current_counts, _ = np.histogram(
        current,
        bins=breakpoints
    )

    # Convert to percentages
    reference_percent = reference_counts / len(reference)
    current_percent = current_counts / len(current)

    # Avoid division by zero
    reference_percent = np.where(
        reference_percent == 0,
        0.0001,
        reference_percent
    )

    current_percent = np.where(
        current_percent == 0,
        0.0001,
        current_percent
    )

    psi = np.sum(
        (current_percent - reference_percent)
        * np.log(current_percent / reference_percent)
    )

    return psi


# --------------------------------------------------
# DRIFT ANALYSIS
# --------------------------------------------------

results = []

for feature in features:

    reference_mean = reference_data[feature].mean()
    current_mean = current_data[feature].mean()

    reference_std = reference_data[feature].std()
    current_std = current_data[feature].std()

    psi = calculate_psi(
        reference_data[feature],
        current_data[feature]
    )

    mean_change_percent = 0

    if reference_mean != 0:
        mean_change_percent = (
            abs(current_mean - reference_mean)
            / abs(reference_mean)
        ) * 100

    # PSI interpretation
    if psi < 0.10:
        drift_status = "No Significant Drift"
    elif psi < 0.25:
        drift_status = "Moderate Drift"
    else:
        drift_status = "Significant Drift"

    results.append({
        "Feature": feature,
        "Reference_Mean": reference_mean,
        "Current_Mean": current_mean,
        "Reference_Std": reference_std,
        "Current_Std": current_std,
        "Mean_Change_Percent": mean_change_percent,
        "PSI": psi,
        "Drift_Status": drift_status
    })

# --------------------------------------------------
# CREATE REPORT
# --------------------------------------------------

report = pd.DataFrame(results)

report = report.sort_values(
    by="PSI",
    ascending=False
)

report.to_csv(
    output_path,
    index=False
)

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n==========================================")
print("DRIFT REPORT")
print("==========================================")

print(
    report[
        [
            "Feature",
            "Mean_Change_Percent",
            "PSI",
            "Drift_Status"
        ]
    ].to_string(index=False)
)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

significant = len(
    report[report["Drift_Status"] == "Significant Drift"]
)

moderate = len(
    report[report["Drift_Status"] == "Moderate Drift"]
)

no_drift = len(
    report[report["Drift_Status"] == "No Significant Drift"]
)

print("\n==========================================")
print("MONITORING SUMMARY")
print("==========================================")

print("Significant drift features :", significant)
print("Moderate drift features    :", moderate)
print("Stable features            :", no_drift)

if significant > 0:

    print("\n⚠️ ALERT:")
    print("Significant data drift detected.")
    print("Model retraining may be required.")

elif moderate > 0:

    print("\n⚠️ WARNING:")
    print("Moderate drift detected.")
    print("Continue monitoring the model.")

else:

    print("\n✅ STATUS:")
    print("No significant data drift detected.")

print("\nDrift report saved to:")
print(output_path)