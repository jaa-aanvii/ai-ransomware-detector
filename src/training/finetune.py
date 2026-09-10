import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.data_pipeline.parser import DatasetParser
from src.data_pipeline.scaler import FeatureScaler
from src.data_pipeline.sequence_builder import build_sequences
from src.training.model import RansomwareLSTM


DATA_PATH = "data/test_sample_concise.xlsx"
BASE_MODEL_PATH = "models/ransomware_lstm.pth"
SCALER_PATH = "models/scaler.pkl"
FINETUNED_MODEL_PATH = "models/ransomware_lstm_finetuned.pth"

SEQUENCE_LENGTH = 6
FEATURE_DIMENSION = 8

BATCH_SIZE = 16
FINETUNE_EPOCHS = 10
FINETUNE_LR = 0.0001

FEATURE_COLUMNS = [
    "write_rate",
    "rename_rate",
    "unique_dirs",
    "files_modified",
    "entropy",
    "dir_dispersion",
    "proc_cpu_pct",
    "handle_count"
]


def main():

    print("Loading telemetry dataset...")

    parser = DatasetParser(DATA_PATH)
    df = parser.load()

    print("Rows:", len(df))

    # ------------------------------------------------
    # Build identical 6 x 8 sequences
    # ------------------------------------------------

    sequences, labels, groups = build_sequences(
        df,
        FEATURE_COLUMNS,
        seq_len=SEQUENCE_LENGTH,
        stride=1
    )

    print("Sequences:", sequences.shape)
    print("Labels:", labels.shape)

    # ------------------------------------------------
    # Load the scaler used by the base model
    # ------------------------------------------------

    scaler = FeatureScaler()
    scaler.load(SCALER_PATH)

    original_shape = sequences.shape

    scaled = scaler.transform(
        sequences.reshape(-1, FEATURE_DIMENSION)
    )

    sequences = scaled.reshape(original_shape)

    # ------------------------------------------------
    # Dataset
    # ------------------------------------------------

    X = torch.tensor(
        sequences,
        dtype=torch.float32
    )

    y = torch.tensor(
        labels,
        dtype=torch.float32
    )

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ------------------------------------------------
    # Device
    # ------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ------------------------------------------------
    # Load BASE MODEL
    # ------------------------------------------------

    model = RansomwareLSTM(
        num_features=FEATURE_DIMENSION,
        hidden_dim=32,
        num_layers=1
    ).to(device)

    model.load_state_dict(
        torch.load(
            BASE_MODEL_PATH,
            map_location=device
        )
    )

    print("Base model loaded:", BASE_MODEL_PATH)

    # ------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FINETUNE_LR
    )

    print()
    print("Starting fine-tuning...")
    print("Learning rate:", FINETUNE_LR)
    print("Epochs:", FINETUNE_EPOCHS)

    model.train()

    for epoch in range(FINETUNE_EPOCHS):

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

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(
            f"Fine-tune Epoch "
            f"{epoch + 1:02d}/{FINETUNE_EPOCHS} "
            f"Loss: {avg_loss:.4f}"
        )

    # ------------------------------------------------
    # Save fine-tuned model
    # ------------------------------------------------

    torch.save(
        model.state_dict(),
        FINETUNED_MODEL_PATH
    )

    print()
    print("Fine-tuning complete.")
    print(
        "Fine-tuned model saved:",
        FINETUNED_MODEL_PATH
    )


if __name__ == "__main__":
    main()