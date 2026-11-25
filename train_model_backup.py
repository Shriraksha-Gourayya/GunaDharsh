
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the CSV
df = pd.read_csv("mango_dataset.csv")

print("Dataset shape:", df.shape)
print("\nLabels:")
print(df["label"].value_counts())

# 2. Select the 18 spectral features
spectral_cols = [
    col for col in df.columns if col.startswith("nm")
]

assert len(spectral_cols) == 18, (
    f"Expected 18 wavelength columns, got {len(spectral_cols)}"
)

# 3. Basic data checks
df["label"] = df["label"].str.strip().str.upper()

if not set(df["label"].unique()).issubset({"RIPE", "ROTTEN"}):
    raise ValueError("Unexpected label found in CSV")

df[spectral_cols] = df[spectral_cols].apply(
    pd.to_numeric, errors="coerce"
)

if df[spectral_cols].isna().any().any():
    raise ValueError("Missing or invalid spectral values found")

# 4. Group consecutive rows into 55-point scans.
# This assumes the CSV is still in original collection order
# and every saved scan contains exactly 55 rows.
if len(df) % 55 != 0:
    raise ValueError(
        "Row count is not divisible by 55. "
        "Check scan completeness before training."
    )

df["scan_id"] = np.arange(len(df)) // 55

X_rows = []
y_scans = []

for scan_id, scan in df.groupby("scan_id", sort=True):
    if len(scan) != 55:
        raise ValueError(f"Scan {scan_id} does not contain 55 rows")

    if scan["label"].nunique() != 1:
        raise ValueError(
            f"Labels are inconsistent within scan {scan_id}"
        )

    # Average and variation for each wavelength
    means = scan[spectral_cols].mean()
    stds = scan[spectral_cols].std(ddof=0)

    features = pd.concat(
        [means.add_suffix("_mean"), stds.add_suffix("_std")]
    )

    X_rows.append(features)
    y_scans.append(scan["label"].iloc[0])

X = pd.DataFrame(X_rows)
y = pd.Series(y_scans, name="label")

print("\nScan-level dataset:", X.shape)
print(y.value_counts())

# 5. Split scans into training and testing sets.
# This is a preliminary evaluation only: repeated scans of the
# same physical mango must not appear in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Create and train the model
model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
)

model.fit(X_train, y_train)

# 7. Evaluate on the held-out scans
predictions = model.predict(X_test)

print("\nAccuracy:", round(
    accuracy_score(y_test, predictions), 4
))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=["RIPE", "ROTTEN"],
    zero_division=0
))

print("Confusion matrix (RIPE, ROTTEN):")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["RIPE", "ROTTEN"]
))

# 8. Save the trained model and feature order
os.makedirs("models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_columns": list(X.columns),
        "wavelength_columns": spectral_cols
    },
    "models/mango_model.joblib"
)

print("\nSaved model to models/mango_model.joblib")
