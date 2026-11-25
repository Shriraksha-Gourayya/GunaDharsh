"""
NutriScan AI - Live Mango Test Collector
Collects one 55-point scan and saves it separately from the training dataset.
Compatible with the current ESP32 serial protocol in mango_sensor.ino.
"""

import csv
import time
from pathlib import Path

import serial
import serial.tools.list_ports

ESP32_PORT = "COM3"       # Change this if your ESP32 uses another port
BAUD_RATE = 115200
EXPECTED_POINTS = 55
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "live_mango_reading.csv"

WAVELENGTHS = [
    410, 435, 460, 485, 510, 535, 560, 585, 610,
    645, 680, 705, 730, 760, 810, 860, 900, 940
]
HEIGHT_LABELS = ["Top", "UpMid", "Center", "LoMid", "Bottom"]
ROTATION_LABELS = [
    "0°", "34°", "68°", "102°", "136°", "170°",
    "136°", "102°", "68°", "34°", "0°"
]
EXPECTED_KEYS = {(p, v) for p in range(11) for v in range(5)}
HEADER = (
    ["label", "mango_id", "rotation_pass", "height_pos"]
    + [f"nm{w}" for w in WAVELENGTHS]
)


def choose_port():
    ports = list(serial.tools.list_ports.comports())
    if ports:
        print("Detected serial ports:")
        for p in ports:
            print(f"  {p.device}: {p.description}")
    else:
        print("No serial ports detected.")
    port = input(f"ESP32 port [{ESP32_PORT}]: ").strip() or ESP32_PORT
    return port


def wait_for_ready(ser, timeout_seconds=15):
    print("\nWaiting for ESP32 READY message...")
    end = time.time() + timeout_seconds
    while time.time() < end:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        if line:
            print(f"ESP32: {line}")
        if "READY" in line:
            return True
    return False


def collect_scan(ser):
    # The current Arduino prints SCANNING MANGO AT ... before each pass.
    # Count markers because the R:index field repeats during the return sweep.
    readings = {}
    current_pass = -1
    scan_started = False
    print("\nStarting scan. Keep the scanner closed and do not touch the mango.\n")
    ser.reset_input_buffer()
    ser.write(b"SCAN\n")
    ser.flush()

    while True:
        raw = ser.readline()
        if not raw:
            print("No serial data received before timeout.")
            break
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        if "FULL MANGO SCAN START" in line:
            scan_started = True

        if "SCANNING MANGO AT ROTATION" in line:
            current_pass += 1
            print(f"Pass {current_pass + 1}/11: {line}")
            continue

        if "SCAN_END" in line:
            print("\nESP32 sent SCAN_END.")
            break

        if not line.startswith("R:"):
            continue

        try:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 20:
                print(f"Skipping malformed row ({len(parts)-2} channels): {line}")
                continue

            rot = int(parts[0].replace("R:", "").strip())
            vert = int(parts[1].replace("V:", "").strip())
            values = [float(x) for x in parts[2:]]
            if len(values) != 18:
                print(f"Skipping row with {len(values)} spectral values.")
                continue
            if not 0 <= current_pass < 11:
                print(f"Skipping reading outside expected pass markers: {line}")
                continue
            if not 0 <= vert < 5:
                print(f"Skipping invalid vertical position: {vert}")
                continue

            key = (current_pass, vert)
            if key in readings:
                print(f"Duplicate pass/height {key}; keeping first reading.")
                continue

            readings[key] = {
                "rot": rot,
                "pass": current_pass,
                "vert": vert,
                "values": values,
            }
            print(
                f"  Saved in memory: pass {current_pass + 1}/11, "
                f"height {HEIGHT_LABELS[vert]}, nm940={values[17]:.4f}"
            )
        except (ValueError, IndexError) as exc:
            print(f"Could not parse row ({exc}): {line}")

    if not scan_started:
        print("\nWARNING: The expected scan-start message was not detected.")
    return readings


def save_scan(readings, mango_id):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Write a new file for each run; do not append to an old test.
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for p in range(11):
            for v in range(5):
                item = readings.get((p, v))
                if item is None:
                    continue
                writer.writerow(
                    ["", mango_id, p, v]
                    + [f"{value:.4f}" for value in item["values"]]
                )
    print(f"\nSaved {len(readings)} rows to:\n{OUTPUT_FILE}")


def main():
    print("=" * 58)
    print("NUTRISCAN AI - LIVE MANGO READING COLLECTOR")
    print("=" * 58)
    mango_id = input("Test mango ID [TEST001]: ").strip() or "TEST001"
    port = choose_port()

    try:
        with serial.Serial(port, BAUD_RATE, timeout=90) as ser:
            # ESP32 resets when the serial port opens; allow it to boot.
            time.sleep(2)
            if not wait_for_ready(ser):
                print(
                    "\nNo READY message detected. Check the port, close Arduino "
                    "Serial Monitor, then rerun. If the board is already ready, "
                    "you may press its RESET button and rerun."
                )
                return

            readings = collect_scan(ser)
            save_scan(readings, mango_id)

            found = set(readings)
            missing = sorted(EXPECTED_KEYS - found)
            if len(readings) != EXPECTED_POINTS or missing:
                print("\nINCOMPLETE SCAN. Do not use this file for model prediction yet.")
                print(f"Expected: {EXPECTED_POINTS} unique points")
                print(f"Received: {len(readings)} unique points")
                if missing:
                    print("Missing pass/height positions:")
                    for p, v in missing:
                        print(
                            f"  Pass {p + 1} ({ROTATION_LABELS[p]}), "
                            f"height {v} ({HEIGHT_LABELS[v]})"
                        )
                print("The partial readings were saved for troubleshooting only.")
            else:
                print("\nCOMPLETE SCAN: all 55 unique positions were collected.")
                print("Next run predict_live_mango.py to test the saved model.")

    except serial.SerialException as exc:
        print(f"\nSerial connection error: {exc}")
        print("Check ESP32_PORT, disconnect Arduino Serial Monitor, and try again.")
    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")


if __name__ == "__main__":
    main()
