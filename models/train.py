import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from lstm_net import RansomwareLSTM

def load_and_preprocess_mlran(data_dir: str, seq_len: int = 6):
    x_train_path = os.path.join(data_dir, "MLRan_X_train_RFE.csv")
    x_test_path = os.path.join(data_dir, "MLRan_X_test_RFE.csv")

    if not os.path.exists(x_train_path) or not os.path.exists(x_test_path):
        raise FileNotFoundError(f"Missing CSV files in {data_dir}.")

    print("Loading MLRan CSV files...")
    train_df = pd.read_csv(x_train_path)
    test_df = pd.read_csv(x_test_path)

    # 1. Extract target ground truth 'sample_type' (0 = Goodware, 1 = Ransomware)
    if 'sample_type' in train_df.columns:
        y_train = train_df['sample_type'].values
        y_test = test_df['sample_type'].values
    else:
        # Fallback if target column is in MLRan_labels.csv
        labels_path = os.path.join(data_dir, "MLRan_labels.csv")
        labels_df = pd.read_csv(labels_path)
        y_train = labels_df['sample_type'].iloc[:len(train_df)].values
        y_test = labels_df['sample_type'].iloc[len(train_df):len(train_df) + len(test_df)].values

    # 2. Drop all non-feature metadata & class label columns
    metadata_cols = ['sample_id', 'sample_type', 'family_label', 'type_label']
    train_df = train_df.drop(columns=[c for c in metadata_cols if c in train_df.columns], errors='ignore')
    test_df = test_df.drop(columns=[c for c in metadata_cols if c in test_df.columns], errors='ignore')

    # Remove any lingering non-numeric columns
    non_num = train_df.select_dtypes(exclude=[np.number]).columns
    if len(non_num) > 0:
        train_df = train_df.drop(columns=non_num)
        test_df = test_df.drop(columns=non_num)

    # Force binary targets strictly to float32 (0.0 or 1.0)
    y_train = np.where(y_train > 0, 1.0, 0.0).astype(np.float32)
    y_test = np.where(y_test > 0, 1.0, 0.0).astype(np.float32)

    X_train_raw = train_df.values
    X_test_raw = test_df.values

    # 3. Scale numerical feature vectors
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    num_features = X_train_scaled.shape[1]
    print(f"Extracted {num_features} dynamic feature indicators for training.")

    # 4. Construct sequential time windows for PyTorch LSTM
    def create_sequences(X_data, y_data):
        sequences, labels = [], []
        for i in range(len(X_data) - seq_len + 1):
            sequences.append(X_data[i : i + seq_len])
            labels.append(y_data[i + seq_len - 1])
        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32).reshape(-1, 1)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test)

    return (
        torch.tensor(X_train_seq), torch.tensor(y_train_seq),
        torch.tensor(X_test_seq), torch.tensor(y_test_seq),
        num_features
    )

def train_mlran_pipeline():
    seq_len = 6
    batch_size = 64
    epochs = 15
    learning_rate = 0.001

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    # Load preprocessed tensors
    X_train, y_train, X_test, y_test, num_features = load_and_preprocess_mlran(data_dir, seq_len=seq_len)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    # Initialize PyTorch LSTM model
    model = RansomwareLSTM(num_features=num_features, hidden_dim=64, num_layers=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\n--- Starting Training Loop ---")
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")

    # Evaluate classification metrics
    print("\n--- Model Evaluation ---")
    model.eval()
    with torch.no_grad():
        raw_preds = model(X_test)
        binary_preds = (raw_preds >= 0.5).float().numpy()
        y_true = y_test.numpy()

    acc = accuracy_score(y_true, binary_preds)
    prec = precision_score(y_true, binary_preds, zero_division=0)
    rec = recall_score(y_true, binary_preds, zero_division=0)
    f1 = f1_score(y_true, binary_preds, zero_division=0)
    cm = confusion_matrix(y_true, binary_preds)

    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"F1-Score : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Negatives: {cm[0][0]} | False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]} | True Positives: {cm[1][1]}")

    # Export ONNX model
    output_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "model.onnx")

    dummy_input = torch.randn(1, seq_len, num_features)
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=14,
        do_constant_folding=True,
        input_names=['telemetry_sequence'], output_names=['risk_score'],
        dynamic_axes={'telemetry_sequence': {0: 'batch_size'}, 'risk_score': {0: 'batch_size'}}
    )
    print(f"\nModel saved successfully to:\n{onnx_path}")

if __name__ == "__main__":
    train_mlran_pipeline()