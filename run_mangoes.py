"""
NutriScan AI: live mango scan + ML prediction.

Save this file in the project root, beside train_model.py and collector_fixed.py.
Run with:
    python run_mango.py

This parser preserves all 55 Arduino measurement rows. The Arduino's reverse
pass revisits the same R/V indices, so duplicate R/V pairs are expected and
must NOT be discarded. A sequence number and scan direction distinguish them.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial
from serial.tools import list_ports


ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "live_testing"
MODEL_PATH = ROOT / "models" / "mango_model.joblib"

LATEST_SCAN = TEST_DIR / "latest_scan.csv"
SENSOR_HISTORY = TEST_DIR / "live_sensor_history.csv"
INCOMPLETE_SCAN = TEST_DIR / "last_incomplete_scan.csv"
LATEST_PREDICTION = TEST_DIR / "latest_prediction.csv"
PREDICTION_HISTORY = TEST_DIR / "predictions_history.csv"
HEATMAP_PATH = TEST_DIR / "latest_prediction_heatmap.png"

BAUD_RATE = 115200
DEFAULT_PORT = "COM3"
EXPECTED_POINTS = 55
WAVELENGTHS = [
    "nm410", "nm435", "nm460", "nm485", "nm510", "nm535",
    "nm560", "nm585", "nm610", "nm645", "nm680", "nm705",
    "nm730", "nm760", "nm810", "nm860", "nm900", "nm940",
]
SCAN_COLUMNS = [
    "timestamp", "mango_id", "measurement_no", "scan_direction",
    "rotation_pass", "height_pos", *WAVELENGTHS,
]


def ask_port() -> str:
    found = list(list_ports.comports())
    if found:
        print("\nAvailable serial ports:")
        for item in found:
            print(f"  {item.device}: {item.description}")
    entered = input(f"\nESP32 serial port [{DEFAULT_PORT}]: ").strip()
    return entered or DEFAULT_PORT


def parse_measurement(line: str):
    """Parse R:<index>,V:<index>,<18 spectral readings>."""
    if not line.startswith("R:"):
        return None
    parts = line.split(",", 2)
    if len(parts) != 3 or not parts[0].startswith("R:") or not parts[1].startswith("V:"):
        return None
    rotation = int(parts[0].split(":", 1)[1].strip())
    height = int(parts[1].split(":", 1)[1].strip())
    values = [float(value.strip()) for value in parts[2].split(",")]
    if len(values) != 18:
        raise ValueError(f"Expected 18 spectral values, got {len(values)}.")
    if not (0 <= rotation <= 10 and 0 <= height <= 4):
        raise ValueError(f"Out-of-range R/V indices: R={rotation}, V={height}.")
    return rotation, height, values


def collect_scan(port: str, mango_id: str) -> pd.DataFrame:
    rows = []
    started = datetime.now().isoformat(timespec="seconds")
    direction = "UNKNOWN"
    clockwise_rows = 0
    anticlockwise_rows = 0
    scan_end = False

    print(f"\nConnecting to {port} at {BAUD_RATE} baud...")
    with serial.Serial(port, BAUD_RATE, timeout=2) as ser:
        # Allow the board to reset after opening the serial port.
        time.sleep(2)
        ser.reset_input_buffer()
        print("Waiting for Arduino READY...")
        ready_until = time.time() + 25
        ready = False
        while time.time() < ready_until:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(line)
                if line.upper() == "READY" or line.upper().endswith(" READY"):
                    ready = True
                    break
        if not ready:
            raise TimeoutError(
                "No READY received from Arduino. Check COM port, USB cable, "
                "and close Arduino Serial Monitor."
            )

        ser.write(b"SCAN\n")
        ser.flush()
        print("\nScan started. Keep the mango and mechanism undisturbed.\n")
        deadline = time.time() + 300

        while time.time() < deadline:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            print(line)
            upper = line.upper()

            if "PHASE 1 - CLOCKWISE" in upper:
                direction = "CLOCKWISE"
                continue
            if "STARTING DIRECT ANTICLOCKWISE MOVEMENT" in upper:
                direction = "ANTICLOCKWISE"
                continue
            if "SCAN_END" in upper:
                scan_end = True
                break

            try:
                parsed = parse_measurement(line)
            except (ValueError, IndexError) as exc:
                print(f"Warning: ignoring malformed sensor row: {exc}")
                continue
            if parsed is None:
                continue

            rotation, height, values = parsed
            if direction == "ANTICLOCKWISE":
                anticlockwise_rows += 1
            else:
                direction = "CLOCKWISE"
                clockwise_rows += 1

            # Do not deduplicate R/V pairs: the Arduino intentionally revisits
            # five physical angles during its anticlockwise phase.
            row = {
                "timestamp": started,
                "mango_id": mango_id,
                "measurement_no": len(rows) + 1,
                "scan_direction": direction,
                "rotation_pass": rotation,
                "height_pos": height,
            }
            row.update(dict(zip(WAVELENGTHS, values)))
            rows.append(row)
            print(f"Accepted measurement {len(rows)}/{EXPECTED_POINTS}")

    df = pd.DataFrame(rows, columns=SCAN_COLUMNS)
    is_complete = (
        scan_end
        and len(df) == EXPECTED_POINTS
        and df[WAVELENGTHS].notna().all().all()
    )
    if not is_complete:
        df.to_csv(INCOMPLETE_SCAN, index=False)
        print(f"\nINCOMPLETE SCAN: received {len(df)}/{EXPECTED_POINTS} measurement rows.")
        print(f"Partial data saved to: {INCOMPLETE_SCAN}")
        print("Latest complete scan and prediction have not been overwritten.")
        raise RuntimeError(f"Scan incomplete: {len(df)}/{EXPECTED_POINTS} measurements.")

    print(
        f"\nValidated {len(df)}/{EXPECTED_POINTS} measurement rows "
        f"({clockwise_rows} clockwise, {anticlockwise_rows} anticlockwise)."
    )
    return df


def append_history(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    add_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=add_header, index=False)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    loaded = joblib.load(MODEL_PATH)
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded["model"], loaded
    return loaded, {}


def decode_label(prediction, bundle) -> str:
    encoder = bundle.get("label_encoder")
    if encoder is not None:
        try:
            return str(encoder.inverse_transform([int(prediction)])[0]).upper()
        except Exception:
            pass

    class_names = bundle.get("class_names")
    if class_names is not None:
        try:
            if isinstance(class_names, dict):
                value = class_names.get(int(prediction), class_names.get(str(int(prediction))))
                if value is not None:
                    return str(value).upper()
            else:
                return str(class_names[int(prediction)]).upper()
        except Exception:
            pass

    # Fallback based on the known training LabelEncoder: RIPE=0, ROTTEN=1.
    try:
        return {0: "RIPE", 1: "ROTTEN"}.get(int(prediction), str(prediction))
    except (TypeError, ValueError):
        return str(prediction)


def predict_latest() -> pd.DataFrame:
    model, bundle = load_model()

    # Read the saved latest scan back from disk before predicting.
    data = pd.read_csv(LATEST_SCAN)
    feature_names = bundle.get("feature_names", WAVELENGTHS)
    if not isinstance(feature_names, (list, tuple, np.ndarray)):
        feature_names = WAVELENGTHS
    features = list(feature_names)
    if not all(name in data.columns for name in features):
        features = WAVELENGTHS

    X = data[features].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Latest scan has missing/non-numeric spectral values; prediction stopped.")

    predictions = model.predict(X)
    result = data.copy()
    result["model_prediction"] = [decode_label(value, bundle) for value in predictions]
    result.to_csv(LATEST_PREDICTION, index=False)
    append_history(result, PREDICTION_HISTORY)
    return result


def make_heatmap(predictions: pd.DataFrame) -> None:
    # The reverse phase revisits physical angles. Display one cell per R/V
    # coordinate by combining repeated predictions: any ROTTEN vote makes the
    # cell ROTTEN. The CSV retains every individual measurement.
    grid_votes = {}
    for _, row in predictions.iterrows():
        key = (int(row["rotation_pass"]), int(row["height_pos"]))
        grid_votes.setdefault(key, []).append(str(row["model_prediction"]).upper())

    grid = np.full((5, 11), np.nan)
    for (rotation, height), labels in grid_votes.items():
        grid[height, rotation] = 1 if any("ROTTEN" in label for label in labels) else 0

    fig, ax = plt.subplots(figsize=(13, 5.5))
    image = ax.imshow(grid, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("NutriScan AI: Live Mango ML Predictions")
    ax.set_xlabel("Arduino rotation index")
    ax.set_ylabel("Vertical position index")
    ax.set_xticks(range(11))
    ax.set_yticks(range(5))
    for height in range(5):
        for rotation in range(11):
            value = grid[height, rotation]
            if not np.isnan(value):
                ax.text(rotation, height, "ROTTEN" if value == 1 else "RIPE",
                        ha="center", va="center", fontsize=7)
    bar = fig.colorbar(image, ax=ax, ticks=[0, 1])
    bar.ax.set_yticklabels(["RIPE", "ROTTEN"])
    fig.tight_layout()
    fig.savefig(HEATMAP_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 58)
    print("       NUTRISCAN AI: LIVE MANGO SCAN + PREDICTION")
    print("=" * 58)

    mango_id = input("\nEnter mango ID (example: LIVE_001): ").strip()
    if not mango_id:
        mango_id = f"LIVE_{datetime.now():%Y%m%d_%H%M%S}"
    port = ask_port()

    try:
        scan = collect_scan(port, mango_id)

        # Only a fully validated 55-row scan replaces the latest snapshot.
        scan.to_csv(LATEST_SCAN, index=False)
        append_history(scan, SENSOR_HISTORY)

        prediction = predict_latest()
        make_heatmap(prediction)

        counts = prediction["model_prediction"].value_counts()
        ripe = int(counts.get("RIPE", 0))
        rotten = int(counts.get("ROTTEN", 0))
        total = len(prediction)

        print("\n" + "=" * 58)
        print("                 LIVE PREDICTION RESULT")
        print("=" * 58)
        print(f"Mango ID          : {mango_id}")
        print(f"Measurements      : {total}/{EXPECTED_POINTS}")
        print(f"RIPE point rows   : {ripe} ({ripe / total * 100:.2f}%)")
        print(f"ROTTEN point rows : {rotten} ({rotten / total * 100:.2f}%)")
        print("\nSaved files:")
        print(f"  Latest scan       : {LATEST_SCAN}")
        print(f"  Sensor history    : {SENSOR_HISTORY}")
        print(f"  Latest prediction : {LATEST_PREDICTION}")
        print(f"  Prediction history: {PREDICTION_HISTORY}")
        print(f"  ML heatmap        : {HEATMAP_PATH}")
        print("\nImportant: predictions are model outputs for scan points, not independently")
        print("verified internal rot. Training labels were generated using an nm940 rule.")
        print("=" * 58)
        return 0

    except serial.SerialException as exc:
        print(f"\nSERIAL ERROR: {exc}")
        print("Close Arduino Serial Monitor or any other program using the COM port.")
        return 1
    except (TimeoutError, FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"\nSTOPPED: {exc}")
        return 1
    except Exception as exc:
        print(f"\nUNEXPECTED ERROR: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
