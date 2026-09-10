import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.data_pipeline.sequence_builder import build_sequences
from src.data_pipeline.scaler import FeatureScaler
from src.training.model import RansomwareLSTM


DATA_PATH = "data/test_sample_concise.xlsx"
SCALER_PATH = "models/scaler.pkl"

BASE_MODEL = "models/ransomware_lstm.pth"
FINETUNED_MODEL = "models/ransomware_lstm_finetuned.pth"

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


def evaluate_model(model_path, X, y):

    model = RansomwareLSTM(
        num_features=8,
        hidden_dim=32,
        num_layers=1
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
        )
    }


def main():

    print("Loading dataset...")

    df = pd.read_excel(
        DATA_PATH,
        sheet_name="sample_windows"
    )

    X, y, groups = build_sequences(
        df,
        FEATURES,
        seq_len=6,
        stride=1
    )

    y = y.flatten().astype(int)

    print("Sequences:", X.shape)
    print("Labels:", y.shape)

    # Load the SAME scaler used during training
    scaler = FeatureScaler()
    scaler.load(SCALER_PATH)

    X_scaled = scaler.transform(
        X.reshape(-1, 8)
    ).reshape(X.shape)

    print()
    print("MODEL COMPARISON")
    print("================")
    print()

    base = evaluate_model(
        BASE_MODEL,
        X_scaled,
        y
    )

    finetuned = evaluate_model(
        FINETUNED_MODEL,
        X_scaled,
        y
    )

    print("BASE MODEL")
    print("----------")
    print(
        f"Accuracy : {base['accuracy']:.4f}"
    )
    print(
        f"Precision: {base['precision']:.4f}"
    )
    print(
        f"Recall   : {base['recall']:.4f}"
    )
    print(
        f"F1 Score : {base['f1']:.4f}"
    )

    print()

    print("FINE-TUNED MODEL")
    print("----------------")
    print(
        f"Accuracy : {finetuned['accuracy']:.4f}"
    )
    print(
        f"Precision: {finetuned['precision']:.4f}"
    )
    print(
        f"Recall   : {finetuned['recall']:.4f}"
    )
    print(
        f"F1 Score : {finetuned['f1']:.4f}"
    )

    print()

    print("CHANGE AFTER FINE-TUNING")
    print("------------------------")
    print(
        f"Accuracy change : "
        f"{finetuned['accuracy'] - base['accuracy']:+.4f}"
    )
    print(
        f"Precision change: "
        f"{finetuned['precision'] - base['precision']:+.4f}"
    )
    print(
        f"Recall change   : "
        f"{finetuned['recall'] - base['recall']:+.4f}"
    )
    print(
        f"F1 change       : "
        f"{finetuned['f1'] - base['f1']:+.4f}"
    )


if __name__ == "__main__":
    main()