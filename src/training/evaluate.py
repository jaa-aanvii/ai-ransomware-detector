import os
import sys

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.data_pipeline.parser import DatasetParser
from src.data_pipeline.scaler import FeatureScaler
from src.training.model import RansomwareLSTM
from src.training.train import create_sequences


DATA_PATH = "data/test_sample_concise.xlsx"
MODEL_PATH = "models/ransomware_lstm.pth"
SCALER_PATH = "models/scaler.pkl"


def main():

    parser = DatasetParser(DATA_PATH)

    df = parser.load()

    sequences, labels = create_sequences(df)

    scaler = FeatureScaler()
    scaler.load(SCALER_PATH)

    original_shape = sequences.shape

    scaled = scaler.transform(
        sequences.reshape(-1, 8)
    )

    sequences = scaled.reshape(
        original_shape
    )

    X = torch.tensor(
        sequences,
        dtype=torch.float32
    )

    model = RansomwareLSTM(
        num_features=8
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    with torch.no_grad():

        probabilities = torch.sigmoid(
            model(X)
        ).numpy().flatten()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print("\nEvaluation Results")
    print("==================")

    print(
        "Accuracy :",
        accuracy_score(labels, predictions)
    )

    print(
        "Precision:",
        precision_score(
            labels,
            predictions,
            zero_division=0
        )
    )

    print(
        "Recall   :",
        recall_score(
            labels,
            predictions,
            zero_division=0
        )
    )

    print(
        "F1 Score :",
        f1_score(
            labels,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            labels,
            predictions
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            labels,
            predictions,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()