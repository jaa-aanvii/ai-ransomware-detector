import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.data_pipeline.parser import DatasetParser
from src.data_pipeline.scaler import FeatureScaler
from src.data_pipeline.window_engine import SlidingWindowEngine
from src.data_pipeline.dataset import RansomwareSequenceDataset
from src.training.model import RansomwareLSTM


DATA_PATH = "data/test_sample_concise.xlsx"
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/ransomware_lstm.pth"

SEQUENCE_LENGTH = 6
FEATURE_DIMENSION = 8

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001


def create_sequences(df):

    sequences = []
    labels = []

    engine = SlidingWindowEngine(
        sequence_length=SEQUENCE_LENGTH,
        feature_dimension=FEATURE_DIMENSION,
        window_size_ms=500
    )

    feature_columns = [
        "write_rate",
        "rename_rate",
        "unique_dirs",
        "files_modified",
        "entropy",
        "dir_dispersion",
        "proc_cpu_pct",
        "handle_count"
    ]

    for _, row in df.iterrows():

        features = row[feature_columns].values.astype(
            np.float32
        )

        result = engine.add_window(
            pid=int(row["pid"]),
            timestamp=float(row["timestamp"]),
            features=features,
            label=int(row["label"])
        )

        if result is not None:

            sequence, label = result

            sequences.append(sequence)
            labels.append(label)

    return np.array(sequences), np.array(labels)


def main():

    print("Loading dataset...")

    parser = DatasetParser(DATA_PATH)

    df = parser.load()

    print("Rows:", len(df))

    # ------------------------------------------------
    # Create sequences
    # ------------------------------------------------

    sequences, labels = create_sequences(df)

    print(
        "Sequences created:",
        sequences.shape
    )

    print(
        "Labels:",
        labels.shape
    )

    if len(sequences) == 0:
        raise RuntimeError(
            "No 6-window sequences could be created."
        )

    # ------------------------------------------------
    # Scale features
    # ------------------------------------------------

    scaler = FeatureScaler()

    flat = sequences.reshape(
        -1,
        FEATURE_DIMENSION
    )

    scaler.fit(flat)

    scaled_flat = scaler.transform(flat)

    scaled_sequences = scaled_flat.reshape(
        sequences.shape
    )

    scaler.save(SCALER_PATH)

    print(
        "Scaler saved:",
        SCALER_PATH
    )

    # ------------------------------------------------
    # Dataset
    # ------------------------------------------------

    dataset = RansomwareSequenceDataset(
        scaled_sequences,
        labels
    )

    # ------------------------------------------------
    # Train / validation split
    # ------------------------------------------------

    train_size = int(
        0.8 * len(dataset)
    )

    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ------------------------------------------------
    # Model
    # ------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    model = RansomwareLSTM(
        num_features=FEATURE_DIMENSION,
        hidden_dim=32,
        num_layers=1
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ------------------------------------------------
    # Training
    # ------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(X)

            loss = criterion(
                output,
                y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        # -----------------------------
        # Validation
        # -----------------------------

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for X, y in val_loader:

                X = X.to(device)
                y = y.to(device)

                output = model(X)

                probability = torch.sigmoid(
                    output
                )

                predictions = (
                    probability >= 0.5
                ).float()

                correct += (
                    predictions == y
                ).sum().item()

                total += y.numel()

        accuracy = correct / total

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"Loss: {total_loss / len(train_loader):.4f} "
            f"Val Accuracy: {accuracy:.4f}"
        )


    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print()
    print("Training complete.")
    print("Model saved:", MODEL_PATH)


if __name__ == "__main__":
    main()