"""
train.py

Replaces the MLRan-based pipeline. MLRan gives one aggregated feature row
per whole sample with no timestamp axis or PID-level time series, so it
cannot be sliced into (PID, 500ms-window) sequences -- there is nothing to
slide a window over. This script instead:

  1. Loads data/telemetry.csv (from generate_telemetry.py) -- real time
     series, one row per (pid, window), correctly ordered and labeled.
  2. Builds (B, T=6, F) sequences strictly within each PID/run
     (sequence_builder.py).
  3. Splits train/test by RUN, not by row, so no process's windows leak
     across the split.
  4. Fits normalization on TRAIN ONLY, saves the stats to disk so Person 2's
     live feature pipeline and Person 3's daemon apply the exact same
     transform at inference time.
  5. Trains RansomwareLSTM, evaluates accuracy / precision / recall / F1 /
     confusion matrix (class-wise, since this is an imbalanced detection
     problem where ransomware-window recall is the number that matters).
  6. Exports model.onnx for onnxruntime, plus a feature_spec.json describing
     input contract (feature order, T, normalization stats) for Person 2/3.
"""

import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from lstm_net import RansomwareLSTM
from sequence_builder import build_sequences, grouped_train_test_split
from generate_telemetry import FEATURE_COLUMNS

SEQ_LEN = 6
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-3
HIDDEN_DIM = 32
NUM_LAYERS = 1
TEST_SIZE = 0.25
SEED = 42

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "telemetry.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "saved_models")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run generate_telemetry.py first to create it."
        )
    return pd.read_csv(path)


def normalize(X_train: np.ndarray, X_test: np.ndarray):
    """Standardize per-feature using TRAIN stats only. X is (N, T, F);
    we compute mean/std over the (N, T) axes for each feature."""
    flat_train = X_train.reshape(-1, X_train.shape[-1])
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0)
    std[std == 0] = 1.0  # avoid divide-by-zero on constant features

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    return X_train_norm.astype(np.float32), X_test_norm.astype(np.float32), mean, std


def train_pipeline():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("--- Loading telemetry ---")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows across {df['run_id'].nunique()} runs.")

    print("\n--- Building sequences (grouped by PID/run, chronological) ---")
    X, y, groups = build_sequences(df, FEATURE_COLUMNS, seq_len=SEQ_LEN, stride=1)
    print(f"Built {len(X)} sequences of shape (T={SEQ_LEN}, F={X.shape[-1]})")

    print("\n--- Splitting train/test by RUN (no PID leakage) ---")
    X_train, y_train, X_test, y_test = grouped_train_test_split(
        X, y, groups, test_size=TEST_SIZE, seed=SEED
    )
    print(f"Train sequences: {len(X_train)}  ({y_train.mean():.3f} positive rate)")
    print(f"Test sequences : {len(X_test)}  ({y_test.mean():.3f} positive rate)")

    print("\n--- Normalizing (fit on train only) ---")
    X_train, X_test, mean, std = normalize(X_train, X_test)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    num_features = X_train.shape[-1]
    model = RansomwareLSTM(num_features=num_features, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Training ---")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:2d}/{EPOCHS}] - Loss: {avg_loss:.4f}")

    print("\n--- Evaluation (held-out runs, never seen in training) ---")
    model.eval()
    with torch.no_grad():
        raw_preds = model(torch.tensor(X_test)).numpy()
    binary_preds = (raw_preds >= 0.5).astype(np.float32)
    y_true = y_test

    acc = accuracy_score(y_true, binary_preds)
    prec = precision_score(y_true, binary_preds, zero_division=0)
    rec = recall_score(y_true, binary_preds, zero_division=0)
    f1 = f1_score(y_true, binary_preds, zero_division=0)
    cm = confusion_matrix(y_true, binary_preds)

    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%   <-- ransomware-window detection rate")
    print(f"F1-Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"                Pred Benign   Pred Ransomware")
    print(f"True Benign     {cm[0][0]:>10d}   {cm[0][1]:>15d}")
    print(f"True Ransomware {cm[1][0]:>10d}   {cm[1][1]:>15d}")
    print("\nClassification report:")
    print(classification_report(y_true, binary_preds, target_names=["benign", "ransomware"], zero_division=0))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save PyTorch weights too, useful for retraining/fine-tuning later
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_state.pt"))

    # Export ONNX for Person 3's onnxruntime daemon
    onnx_path = os.path.join(OUTPUT_DIR, "model.onnx")
    dummy_input = torch.randn(1, SEQ_LEN, num_features)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=17,
        do_constant_folding=True,
        input_names=["telemetry_sequence"], output_names=["risk_score"],
        dynamic_axes={"telemetry_sequence": {0: "batch_size"}, "risk_score": {0: "batch_size"}},
    )
    print(f"\nSaved PyTorch weights -> {os.path.join(OUTPUT_DIR, 'model_state.pt')}")
    print(f"Exported ONNX model  -> {onnx_path}")

    # Save the exact interface contract Person 2 and Person 3 need
    feature_spec = {
        "feature_order": FEATURE_COLUMNS,
        "sequence_length": SEQ_LEN,
        "window_ms": 500,
        "normalization": {
            "method": "standard",
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "decision_threshold": 0.85,
        "sustained_windows_required": 2,
        "notes": (
            "Apply (x - mean) / std per feature, in this exact order, to each "
            "500ms window BEFORE stacking into a (1, 6, F) sequence for "
            "onnxruntime inference. Sequences must never mix windows from "
            "different PIDs."
        ),
    }
    spec_path = os.path.join(OUTPUT_DIR, "feature_spec.json")
    with open(spec_path, "w") as f:
        json.dump(feature_spec, f, indent=2)
    print(f"Saved interface spec  -> {spec_path}")


if __name__ == "__main__":
    train_pipeline()
