import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# PROJECT PATH
# ============================================================

project_root = Path(__file__).resolve().parent.parent

data_path = (
    project_root
    / "data"
    / "processed"
    / "ml_dataset.csv"
)


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

target = "Condition_Label"


# ============================================================
# LOAD DATA
# ============================================================

print("\n==========================================")
print("5-FOLD CROSS-VALIDATION")
print("==========================================")

df = pd.read_csv(data_path)

print("\nDataset shape:", df.shape)


X = df[features]
y = df[target]


print("\nFeature shape:", X.shape)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# STRATIFIED K-FOLD
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []


# ============================================================
# CROSS-VALIDATION
# ============================================================

print("\n==========================================")
print("TRAINING 5 FOLDS")
print("==========================================")

for fold, (train_index, val_index) in enumerate(
    skf.split(X, y), start=1
):

    X_train = X.iloc[train_index]
    X_val = X.iloc[val_index]

    y_train = y.iloc[train_index]
    y_val = y.iloc[val_index]


    # --------------------------------------------------------
    # SCALING
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_val_scaled = scaler.transform(X_val)


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )


    model.fit(
        X_train_scaled,
        y_train
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    y_pred = model.predict(
        X_val_scaled
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0
    )


    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)


    print(
        f"\nFold {fold}"
    )

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
# FINAL RESULTS
# ============================================================

print("\n==========================================")
print("CROSS-VALIDATION RESULTS")
print("==========================================")


print(
    "\nAccuracy:"
)

print(
    "Mean : %.4f"
    % np.mean(accuracy_scores)
)

print(
    "Std  : %.4f"
    % np.std(accuracy_scores)
)


print(
    "\nPrecision:"
)

print(
    "Mean : %.4f"
    % np.mean(precision_scores)
)

print(
    "Std  : %.4f"
    % np.std(precision_scores)
)


print(
    "\nRecall:"
)

print(
    "Mean : %.4f"
    % np.mean(recall_scores)
)

print(
    "Std  : %.4f"
    % np.std(recall_scores)
)


print(
    "\nF1 Score:"
)

print(
    "Mean : %.4f"
    % np.mean(f1_scores)
)

print(
    "Std  : %.4f"
    % np.std(f1_scores)
)


# ============================================================
# PERCENTAGE RESULTS
# ============================================================

print("\n==========================================")
print("FINAL 5-FOLD SUMMARY")
print("==========================================")

print(
    "Mean Accuracy : %.2f %%"
    % (np.mean(accuracy_scores) * 100)
)

print(
    "Mean Precision: %.2f %%"
    % (np.mean(precision_scores) * 100)
)

print(
    "Mean Recall   : %.2f %%"
    % (np.mean(recall_scores) * 100)
)

print(
    "Mean F1 Score : %.2f %%"
    % (np.mean(f1_scores) * 100)
)


print("\n==========================================")
print("5-FOLD CROSS-VALIDATION COMPLETED")
print("==========================================")