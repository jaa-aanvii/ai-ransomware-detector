"""
models/train.py

Same training pipeline as before -- only the imports and file paths
changed to match the new package layout:

    project_root/
      models/
        lstm_net.py
        train.py          <- this file
      src/
        data_pipeline/
          generate_telemetry.py
          sequence_builder.py
      tests/
        test_model_lstm.py
      data/                <- generated telemetry lands here
      saved_models/        <- trained model + spec land here

Run from the project root as a module, NOT as a plain script, so the
package imports resolve correctly:

    python -m models.train

(Running `python models/train.py` directly will fail with an import
error, since Python won't know where the `src` package is without the
project root being treated as the top-level package context.)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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

from models.lstm_net import RansomwareLSTM
from src.data_pipeline.sequence_builder import build_sequences, grouped_train_test_split
from src.data_pipeline.generate_telemetry import FEATURE_COLUMNS

SEQ_LEN = 6
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-3
HIDDEN_DIM = 32
NUM_LAYERS = 1
TEST_SIZE = 0.25
SEED = 42

# This file now lives at <project_root>/models/train.py, so go up ONE
# level to reach the project root, then down into data/ and saved_models/.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "telemetry.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "saved_models")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data_pipeline.generate_telemetry` first to create it."
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
