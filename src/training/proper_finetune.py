import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_pipeline.sequence_builder import build_sequences
from src.data_pipeline.scaler import FeatureScaler
from src.training.model import RansomwareLSTM


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/test_sample_concise.xlsx"

BASE_MODEL_PATH = "models/proper_base_lstm.pth"
FINETUNED_MODEL_PATH = "models/proper_finetuned_lstm.pth"
SCALER_PATH = "models/proper_scaler.pkl"

SEQUENCE_LENGTH = 6
FEATURE_DIMENSION = 8

BATCH_SIZE = 8

BASE_EPOCHS = 40
BASE_LR = 0.001

FINETUNE_EPOCHS = 15
FINETUNE_LR = 0.0001

BASE_RUNS = [0, 1, 130]
FINETUNE_RUNS = [131]
TEST_RUNS = [132]

FEATURES = [
    "write_rate",
    "rename_rate",
    "unique_dirs",
    "files_modified",
    "entropy",
    "dir_dispersion",
    "proc_cpu_pct",
    "handle_count"
]


# ============================================================
# SEQUENCE CREATION
# ============================================================

def make_sequences(df):

    X, y, groups = build_sequences(
        df,
        FEATURES,
        seq_len=SEQUENCE_LENGTH,
        stride=1
    )

    return X, y.flatten()


# ============================================================
# SCALING
# ============================================================

def scale_sequences(X, scaler):

    original_shape = X.shape

    X_scaled = scaler.transform(
        X.reshape(-1, FEATURE_DIMENSION)
    )

    return X_scaled.reshape(original_shape)


# ============================================================
# TRAIN BASE MODEL
# ============================================================

def train_base_model(X, y, device):

    print("\n================================")
    print("BASE MODEL TRAINING")
    print("================================")

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32
    ).unsqueeze(1)

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = RansomwareLSTM(
        num_features=FEATURE_DIMENSION,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3
    ).to(device)

    # Class weighting
    positive = np.sum(y == 1)
    negative = np.sum(y == 0)

    pos_weight = torch.tensor(
        [negative / positive],
        dtype=torch.float32,
        device=device
    )

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=BASE_LR,
        weight_decay=1e-5
    )

    for epoch in range(BASE_EPOCHS):

        model.train()

        total_loss = 0.0

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            output = model(batch_X)

            loss = criterion(
                output,
                batch_y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(
            f"Base Epoch "
            f"{epoch + 1:02d}/{BASE_EPOCHS} "
            f"Loss: {avg_loss:.4f}"
        )

    torch.save(
        model.state_dict(),
        BASE_MODEL_PATH
    )

    print(
        f"\nBase model saved: "
        f"{BASE_MODEL_PATH}"
    )


# ============================================================
# FINE-TUNE MODEL
# ============================================================

def finetune_model(X, y, device):

    print("\n================================")
    print("FINE-TUNING")
    print("================================")

    model = RansomwareLSTM(
        num_features=FEATURE_DIMENSION,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3
    ).to(device)

    model.load_state_dict(
        torch.load(
            BASE_MODEL_PATH,
            map_location=device
        )
    )

    print(
        f"Loaded base model: "
        f"{BASE_MODEL_PATH}"
    )

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32
    ).unsqueeze(1)

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    positive = np.sum(y == 1)
    negative = np.sum(y == 0)

    pos_weight = torch.tensor(
        [negative / positive],
        dtype=torch.float32,
        device=device
    )

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FINETUNE_LR,
        weight_decay=1e-5
    )

    for epoch in range(FINETUNE_EPOCHS):

        model.train()

        total_loss = 0.0

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            output = model(batch_X)

            loss = criterion(
                output,
                batch_y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(
            f"Fine-tune Epoch "
            f"{epoch + 1:02d}/{FINETUNE_EPOCHS} "
            f"Loss: {avg_loss:.4f}"
        )

    torch.save(
        model.state_dict(),
        FINETUNED_MODEL_PATH
    )

    print(
        f"\nFine-tuned model saved: "
        f"{FINETUNED_MODEL_PATH}"
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model_path, X, y):

    model = RansomwareLSTM(
        num_features=FEATURE_DIMENSION,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu"
        )
    )

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    with torch.no_grad():

        probabilities = torch.sigmoid(
            model(X_tensor)
        ).numpy().flatten()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return probabilities, predictions


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y, predictions):

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )

    return {
        "accuracy": accuracy_score(
            y,
            predictions
        ),

        "precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y,
            predictions,
            zero_division=0
        ),

        "confusion": confusion_matrix(
            y,
            predictions
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading dataset...")

    df = pd.read_excel(
        DATA_PATH,
        sheet_name="sample_windows"
    )

    print(
        f"Total rows: {len(df)}"
    )

    print(
        f"\nBase runs: {BASE_RUNS}"
    )

    print(
        f"Fine-tune runs: {FINETUNE_RUNS}"
    )

    print(
        f"Test runs: {TEST_RUNS}"
    )

    # --------------------------------------------------------
    # SPLIT RUNS
    # --------------------------------------------------------

    base_df = df[
        df["run_id"].isin(BASE_RUNS)
    ].copy()

    finetune_df = df[
        df["run_id"].isin(FINETUNE_RUNS)
    ].copy()

    test_df = df[
        df["run_id"].isin(TEST_RUNS)
    ].copy()

    print(
        f"\nBase rows: {len(base_df)}"
    )

    print(
        f"Fine-tune rows: {len(finetune_df)}"
    )

    print(
        f"Test rows: {len(test_df)}"
    )

    # --------------------------------------------------------
    # BUILD SEQUENCES
    # --------------------------------------------------------

    X_base, y_base = make_sequences(
        base_df
    )

    X_finetune, y_finetune = make_sequences(
        finetune_df
    )

    X_test, y_test = make_sequences(
        test_df
    )

    print(
        f"\nBase sequences: {X_base.shape}"
    )

    print(
        f"Fine-tune sequences: {X_finetune.shape}"
    )

    print(
        f"Test sequences: {X_test.shape}"
    )

    # --------------------------------------------------------
    # SCALER
    # --------------------------------------------------------

    scaler = FeatureScaler()

    scaler.fit(
        X_base.reshape(
            -1,
            FEATURE_DIMENSION
        )
    )

    scaler.save(
        SCALER_PATH
    )

    print(
        f"\nScaler saved: {SCALER_PATH}"
    )

    # IMPORTANT:
    # scaler is fitted ONLY on base training data.

    X_base = scale_sequences(
        X_base,
        scaler
    )

    X_finetune = scale_sequences(
        X_finetune,
        scaler
    )

    X_test = scale_sequences(
        X_test,
        scaler
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # TRAIN BASE
    # --------------------------------------------------------

    train_base_model(
        X_base,
        y_base,
        device
    )

    # --------------------------------------------------------
    # BASE MODEL TEST
    # --------------------------------------------------------

    _, base_predictions = evaluate_model(
        BASE_MODEL_PATH,
        X_test,
        y_test
    )

    base_results = calculate_metrics(
        y_test,
        base_predictions
    )

    # --------------------------------------------------------
    # FINE-TUNE
    # --------------------------------------------------------

    finetune_model(
        X_finetune,
        y_finetune,
        device
    )

    # --------------------------------------------------------
    # FINE-TUNED MODEL TEST
    # --------------------------------------------------------

    _, finetuned_predictions = evaluate_model(
        FINETUNED_MODEL_PATH,
        X_test,
        y_test
    )

    finetuned_results = calculate_metrics(
        y_test,
        finetuned_predictions
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n================================")
    print("UNSEEN TEST RESULTS")
    print("================================")

    print("\nBASE MODEL")
    print("--------------------------------")

    print(
        f"Accuracy : "
        f"{base_results['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{base_results['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{base_results['recall']:.4f}"
    )

    print(
        f"F1 Score : "
        f"{base_results['f1']:.4f}"
    )

    print("Confusion Matrix:")

    print(
        base_results["confusion"]
    )

    print("\nFINE-TUNED MODEL")
    print("--------------------------------")

    print(
        f"Accuracy : "
        f"{finetuned_results['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{finetuned_results['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{finetuned_results['recall']:.4f}"
    )

    print(
        f"F1 Score : "
        f"{finetuned_results['f1']:.4f}"
    )

    print("Confusion Matrix:")

    print(
        finetuned_results["confusion"]
    )

    print("\n================================")
    print("IMPROVEMENT")
    print("================================")

    print(
        f"Accuracy change : "
        f"{finetuned_results['accuracy'] - base_results['accuracy']:+.4f}"
    )

    print(
        f"Precision change: "
        f"{finetuned_results['precision'] - base_results['precision']:+.4f}"
    )

    print(
        f"Recall change   : "
        f"{finetuned_results['recall'] - base_results['recall']:+.4f}"
    )

    print(
        f"F1 change       : "
        f"{finetuned_results['f1'] - base_results['f1']:+.4f}"
    )


if __name__ == "__main__":
    main()