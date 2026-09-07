"""
test_model.py

ONE test script, using the SAME generate_telemetry.py your model was
trained on -- no second generator file needed. The only difference
between "training data" and "test data" here is the random seed:

  - train.py calls generate_telemetry.generate_dataset() with the
    module's default seed (42) -> what the model learned from.
  - This script calls generate_telemetry.set_seed(TEST_SEED) with a
    DIFFERENT seed, then generates a brand-new batch of runs. Same
    realistic patterns (benign / aggressive-benign / ransomware), but
    every single number is freshly drawn -- nothing here was seen
    during training, and nothing is memorized.

What this checks: whether the model generalizes to new random draws
from the same behavioral patterns, not whether it memorized specific
training rows. It will NOT catch cases where a real ransomware/benign
sample behaves in a way generate_telemetry.py doesn't model at all --
that gap can only be closed once real captured telemetry exists.

Steps:
  1. Generate a fresh test batch with a new seed (never touches
     data/telemetry.csv, the training file).
  2. Build (T, F) sequences the same way sequence_builder.py always does.
  3. Load the FROZEN normalization stats + feature order from
     saved_models/feature_spec.json (never refit on this new data --
     a live daemon can't peek at future data either).
  4. Run inference through the ACTUAL saved model.onnx via onnxruntime
     (same engine Person 3's daemon uses).
  5. Print accuracy / precision / recall / F1 / confusion matrix.
  6. Save a readable sample of the generated test data to an .xlsx file
     so you can eyeball the actual numbers yourself.

Run train.py once first so saved_models/ exists, then run this.
"""

import os
import json

import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

import generate_telemetry as gt
from sequence_builder import build_sequences

TEST_SEED = 999  # anything different from the training seed (42)
N_BENIGN = 90
N_AGGRESSIVE_BENIGN = 40
N_RANSOMWARE = 90
# These two are the actual stress test: harder variants NOT emphasized in
# training, to check generalization beyond "same pattern, different seed"
# (which -- as you'll see if you set these to 0 -- still scores ~100% and
# proves nothing).
N_STEALTHY_RANSOMWARE = 60
N_BURSTY_BENIGN = 60

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "model.onnx")
SPEC_PATH = os.path.join(BASE_DIR, "saved_models", "feature_spec.json")
EXCEL_SAMPLE_PATH = os.path.join(BASE_DIR, "test_sample.xlsx")


def load_spec(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run train.py first.")
    with open(path) as f:
        return json.load(f)


def save_excel_sample(df: pd.DataFrame, path: str, rows_per_label: int = 40):
    """Concise, readable sample of the generated test data -- not the
    full multi-thousand-row set, just enough per class to inspect by
    eye: a few full runs of each type, in order, with the columns you
    actually care about."""
    picked_runs = []
    for label in (0, 1):
        run_ids = df.loc[df["label"] == label, "run_id"].unique()
        picked_runs.extend(run_ids[:3])  # 3 example runs per class

    sample = df[df["run_id"].isin(picked_runs)].copy()
    sample = sample.sort_values(["run_id", "window_index"])
    sample = sample.head(rows_per_label * 2 * 3)  # keep the file short

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sample.to_excel(writer, sheet_name="sample_windows", index=False)

        summary = df.groupby("label")[gt.FEATURE_COLUMNS].mean().reset_index()
        summary["label"] = summary["label"].map({0: "benign", 1: "ransomware"})
        summary.to_excel(writer, sheet_name="class_averages", index=False)

    print(f"Saved readable test sample -> {path}")


def main():
    spec = load_spec(SPEC_PATH)
    feature_order = spec["feature_order"]
    seq_len = spec["sequence_length"]
    mean = np.array(spec["normalization"]["mean"], dtype=np.float32)
    std = np.array(spec["normalization"]["std"], dtype=np.float32)

    print(f"--- Generating a FRESH batch (seed={TEST_SEED}, never used in training) ---")
    gt.set_seed(TEST_SEED)
    df = gt.generate_dataset(
        n_benign_runs=N_BENIGN,
        n_aggressive_benign_runs=N_AGGRESSIVE_BENIGN,
        n_ransomware_runs=N_RANSOMWARE,
        n_stealthy_ransomware_runs=N_STEALTHY_RANSOMWARE,
        n_bursty_benign_runs=N_BURSTY_BENIGN,
    )
    print(f"{len(df)} rows across {df['run_id'].nunique()} runs. "
          f"Positive rate: {df['label'].mean():.3f}")

    save_excel_sample(df, EXCEL_SAMPLE_PATH)

    print("\n--- Building sequences ---")
    X, y, groups = build_sequences(df, feature_order, seq_len=seq_len, stride=1)
    print(f"Built {len(X)} sequences. Positive rate: {y.mean():.3f}")

    print("\n--- Applying FROZEN normalization from training (no refitting) ---")
    X_norm = ((X - mean) / std).astype(np.float32)

    print(f"\n--- Running inference via onnxruntime: {MODEL_PATH} ---")
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    raw_preds = session.run(None, {input_name: X_norm})[0]
    binary_preds = (raw_preds >= 0.5).astype(np.float32)

    acc = accuracy_score(y, binary_preds)
    prec = precision_score(y, binary_preds, zero_division=0)
    rec = recall_score(y, binary_preds, zero_division=0)
    f1 = f1_score(y, binary_preds, zero_division=0)
    cm = confusion_matrix(y, binary_preds)

    print("\n=== RESULTS ON FRESH, UNSEEN DATA ===")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%   <-- ransomware-window detection rate")
    print(f"F1-Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"                Pred Benign   Pred Ransomware")
    print(f"True Benign     {cm[0][0]:>10d}   {cm[0][1]:>15d}")
    print(f"True Ransomware {cm[1][0]:>10d}   {cm[1][1]:>15d}")
    print("\nClassification report:")
    print(classification_report(y, binary_preds, target_names=["benign", "ransomware"], zero_division=0))


if __name__ == "__main__":
    main()
