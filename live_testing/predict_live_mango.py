"""
NutriScan AI - Live Mango Model Test
Loads live_mango_reading.csv, validates the 55-point scan, and predicts each point.
The output is a model prediction, not independently verified fruit ground truth.
"""

import csv
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
INPUT_FILE = BASE_DIR / "live_mango_reading.csv"
OUTPUT_FILE = BASE_DIR / "live_mango_prediction.csv"
MODEL_FILE = PROJECT_DIR / "models" / "mango_model.joblib"

FEATURES = [
    "nm410", "nm435", "nm460", "nm485", "nm510", "nm535",
    "nm560", "nm585", "nm610", "nm645", "nm680", "nm705",
    "nm730", "nm760", "nm810", "nm860", "nm900", "nm940",
]
EXPECTED_POINTS = 55


def extract_model(loaded):
    """Support a saved sklearn estimator or a common dictionary model bundle."""
    if hasattr(loaded, "predict"):
        return loaded

    if isinstance(loaded, dict):
        for key in ("model", "best_model", "pipeline", "estimator", "classifier"):
            candidate = loaded.get(key)
            if hasattr(candidate, "predict"):
                return candidate
        # Some bundles store the estimator under a nonstandard key.
        for value in loaded.values():
            if hasattr(value, "predict"):
                return value

    raise TypeError(
        "Could not find a scikit-learn model/pipeline inside mango_model.joblib. "
        "Please share the final joblib.dump(...) section of train_model.py."
    )


def main():
    print("=" * 58)
    print("NUTRISCAN AI - LIVE MANGO PREDICTION")
    print("=" * 58)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Live reading file not found: {INPUT_FILE}\n"
            "Run collect_live_mango.py first."
        )
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}\n"
            "Train the model first and confirm it is saved at that location."
        )

    df = pd.read_csv(INPUT_FILE)
    required = ["mango_id", "rotation_pass", "height_pos"] + FEATURES
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    if df[FEATURES].isna().any().any():
        bad_rows = df.index[df[FEATURES].isna().any(axis=1)].tolist()
        raise ValueError(f"Missing spectral values found in CSV rows: {bad_rows[:10]}")

    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if df[FEATURES].isna().any().any():
        raise ValueError("Some spectral values are not numeric.")
    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Infinite or non-finite spectral values found.")

    # Verify unique scan positions, not just row count.
    if len(df) != EXPECTED_POINTS:
        raise ValueError(
            f"Incomplete scan: expected 55 rows, found {len(df)}. "
            "Rescan before predicting."
        )
    keys = list(zip(df["rotation_pass"], df["height_pos"]))
    if len(set(keys)) != EXPECTED_POINTS:
        raise ValueError("Duplicate scan positions found; rescan before predicting.")
    expected = {(p, v) for p in range(11) for v in range(5)}
    actual = {(int(p), int(v)) for p, v in keys}
    if actual != expected:
        missing = sorted(expected - actual)
        raise ValueError(f"Scan positions are incomplete. Missing: {missing}")

    loaded = joblib.load(MODEL_FILE)
    model = extract_model(loaded)

    X = df[FEATURES]
    predictions = model.predict(X)

    # If predict_proba is available, save the probability for the predicted class.
    confidence = []
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X)
            classes = list(model.classes_)
            for pred, row in zip(predictions, probabilities):
                confidence.append(float(row[classes.index(pred)]))
        except (AttributeError, ValueError, IndexError):
            confidence = []

    result = df[["mango_id", "rotation_pass", "height_pos"]].copy()
    result["model_prediction"] = predictions
    if confidence:
        result["predicted_class_probability"] = confidence
    result.to_csv(OUTPUT_FILE, index=False)

    print(f"\nMango ID: {df['mango_id'].iloc[0]}")
    print(f"Points predicted: {len(result)}/55")
    print("\nPoint-level prediction counts:")
    print(result["model_prediction"].value_counts(dropna=False).to_string())
    print(f"\nSaved point-level results to:\n{OUTPUT_FILE}")
    print(
        "\nImportant: these are predictions from the trained model. Because the "
        "training labels were generated using the nm940 threshold and nm940 is "
        "also an input feature, this test does not independently prove actual "
        "ripeness or internal rot."
    )


if __name__ == "__main__":
    main()
