"""
ensemble.py -- the full pipeline requested:

    1. Train LSTM, GRU, and BiLSTM independently (each with its own
       output layer, so each actually learns something on its own).
    2. Remove each model's output layer (return_embedding=True) and
       run every sample through all three to get three embeddings.
    3. Concatenate the three embeddings into one feature vector per
       sample.
    4. Apply PCA to the concatenated embedding -- this both reduces
       dimensionality and removes redundant/duplicated signal, since
       PCA's components are by construction uncorrelated with each
       other (three separately-trained encoders will likely learn some
       overlapping representation of the same underlying features, and
       PCA collapses that overlap rather than treating it as extra
       information).
    5. Train a final small "fine-tuning" classifier on the PCA-reduced
       fused features and produce the final ransomware/benign output.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.mlran_ensemble.models import MODEL_REGISTRY

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64


def train_full_model(model_name, params, X_train, y_train, epochs=30, device=DEVICE):
    """Trains one encoder end-to-end (WITH its output layer) so it
    learns a useful representation before being reduced to a feature
    extractor."""
    input_size = X_train.shape[2]
    model_cls = MODEL_REGISTRY[model_name]

    model = model_cls(
        input_size=input_size,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    # Mini-batch training -- see the note in tune.py's _train_and_score;
    # same reasoning applies here for the final, longer training run.
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def extract_embedding(model, X, device=DEVICE, batch_size=256):
    """Runs X through `model` with its output layer skipped
    (return_embedding=True) and returns a numpy array of embeddings."""
    model.eval()
    embeddings = []

    for start in range(0, len(X), batch_size):
        batch = torch.tensor(X[start:start + batch_size], dtype=torch.float32).to(device)
        emb = model(batch, return_embedding=True)
        embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


class FineTuningMLP(nn.Module):
    """The 'fine-tuning model' that consumes the PCA-reduced, fused
    embedding from all three encoders and produces the final output."""

    def __init__(self, input_dim, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class EnsembleResult:
    def __init__(self, encoders, pca, final_model, metrics):
        self.encoders = encoders
        self.pca = pca
        self.final_model = final_model
        self.metrics = metrics


def run_ensemble_pipeline(
    X_train,
    y_train,
    X_test,
    y_test,
    tuned_params,
    encoder_epochs=30,
    pca_variance=0.95,
    final_epochs=60,
    final_lr=1e-3,
    device=DEVICE,
):
    """
    tuned_params: dict like {"lstm": {...}, "gru": {...}, "bilstm": {...}}
    as produced by tune.py's search functions.
    """
    encoders = {}
    for name, params in tuned_params.items():
        encoders[name] = train_full_model(name, params, X_train, y_train, encoder_epochs, device)

    # Step 2 + 3: strip output layers, extract + concatenate embeddings.
    def fused_embedding(X):
        parts = [extract_embedding(encoders[name], X, device) for name in ("lstm", "gru", "bilstm")]
        return np.concatenate(parts, axis=1)

    fused_train = fused_embedding(X_train)
    fused_test = fused_embedding(X_test)

    # Step 4: PCA -- reduce dimensionality and collapse redundant signal
    # shared across the three encoders. n_components chosen to retain
    # `pca_variance` of total variance rather than a fixed count, since
    # the "right" amount of redundancy to remove isn't known up front.
    pca = PCA(n_components=pca_variance, svd_solver="full")
    reduced_train = pca.fit_transform(fused_train)
    reduced_test = pca.transform(fused_test)

    # Step 5: train the fine-tuning model on the fused, reduced features.
    final_model = FineTuningMLP(input_dim=reduced_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=final_lr)
    criterion = nn.BCEWithLogitsLoss()

    ft_ds = TensorDataset(
        torch.tensor(reduced_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    ft_loader = DataLoader(ft_ds, batch_size=BATCH_SIZE, shuffle=True)

    final_model.train()
    for epoch in range(final_epochs):
        for X_batch, y_batch in ft_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = final_model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluation on MLRan's held-out test split.
    final_model.eval()
    with torch.no_grad():
        test_logits = final_model(torch.tensor(reduced_test, dtype=torch.float32).to(device))
        test_probs = torch.sigmoid(test_logits).cpu().numpy().ravel()

    test_preds = (test_probs >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, test_preds)),
        "precision": float(precision_score(y_test, test_preds, zero_division=0)),
        "recall": float(recall_score(y_test, test_preds, zero_division=0)),
        "f1": float(f1_score(y_test, test_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_probs)),
        "confusion_matrix": [[int(v) for v in row] for row in confusion_matrix(y_test, test_preds)],
        "classification_report": classification_report(y_test, test_preds, zero_division=0),
        "pca_components_kept": int(pca.n_components_),
        "pca_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
        "fused_embedding_dim_before_pca": int(fused_train.shape[1]),
    }

    return EnsembleResult(encoders, pca, final_model, metrics)
