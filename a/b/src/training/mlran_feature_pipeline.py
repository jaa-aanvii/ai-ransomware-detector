"""
mlran_feature_pipeline.py

    MLRan CSV
      -> remove metadata/labels
      -> 483 features
      -> StandardScaler (loaded from train_mlran_extractors.py -- fit on
         training data only, reused here, never refit)
      -> pretrained GRU + LSTM + BiLSTM extractors (weights loaded from
         models/mlran/*_extractor.pth, produced by
         train_mlran_extractors.py -- run in eval mode, no gradients)
      -> concatenate -> 512-dim fused representation
      -> PCA fit on TRAINING fused representations only
      -> transform test fused representations with that fitted PCA
      -> save train/test PCA features + labels for the fine-tuning stage

This script does not train anything and does not touch test labels for
any fitting decision -- it only runs inference with already-trained,
already-saved weights and a transform-only PCA on the test split.

Run train_mlran_extractors.py first if models/mlran/*_extractor.pth
don't exist yet.
"""

import argparse
import os
import pickle

import numpy as np
import torch
import sys 
from pathlib import Path
from sklearn.decomposition import PCA

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.mlran_data import load_raw, load_scaler, to_model_input
from src.training.mlran_models import (
    BiLSTMFeatureExtractor,
    GRUFeatureExtractor,
    LSTMFeatureExtractor,
)

MODEL_DIR = "models/mlran"
OUTPUT_DIR = "data/MLRan_processed"


def load_pretrained_extractors(model_dir, device):
    gru = GRUFeatureExtractor(input_dim=483, hidden_dim=128, num_layers=2, dropout=0.3)
    lstm = LSTMFeatureExtractor(input_dim=483, hidden_dim=128, num_layers=2, dropout=0.3)
    bilstm = BiLSTMFeatureExtractor(input_dim=483, hidden_dim=128, num_layers=2, dropout=0.3)

    gru.load_state_dict(torch.load(os.path.join(model_dir, "gru_extractor.pth"), map_location=device))
    lstm.load_state_dict(torch.load(os.path.join(model_dir, "lstm_extractor.pth"), map_location=device))
    bilstm.load_state_dict(
        torch.load(os.path.join(model_dir, "bilstm_extractor.pth"), map_location=device)
    )

    for m in (gru, lstm, bilstm):
        m.to(device)
        m.eval()

    return gru, lstm, bilstm


@torch.no_grad()
def extract_fused_features(gru, lstm, bilstm, X_seq, device, batch_size=256):
    """Runs the three PRETRAINED, FROZEN extractors over X_seq and
    concatenates their outputs into one 512-dim vector per sample.
    No gradients, no training -- inference only."""
    all_fused = []

    for start in range(0, len(X_seq), batch_size):
        batch = torch.tensor(X_seq[start:start + batch_size], dtype=torch.float32).to(device)

        gru_feat = gru(batch)
        lstm_feat = lstm(batch)
        bilstm_feat = bilstm(batch)

        fused = torch.cat([gru_feat, lstm_feat, bilstm_feat], dim=1)
        all_fused.append(fused.cpu().numpy())

    return np.concatenate(all_fused, axis=0)


def main():
    parser = argparse.ArgumentParser(description="MLRan feature extraction -> fusion -> PCA pipeline")
    parser.add_argument("--data-dir", default="data/MLRan")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--pca-variance", type=float, default=0.95,
        help="Fraction of variance PCA should retain (dimensionality picked automatically)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    missing = [
        f for f in ("gru_extractor.pth", "lstm_extractor.pth", "bilstm_extractor.pth", "scaler.pkl")
        if not os.path.exists(os.path.join(args.model_dir, f))
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing pretrained artifacts in {args.model_dir}: {missing}. "
            f"Run train_mlran_extractors.py first."
        )

    print("Loading MLRan train/test split...")
    X_train_raw, y_train, X_test_raw, y_test, feature_cols = load_raw(args.data_dir)
    print(f"  train: {X_train_raw.shape}, test: {X_test_raw.shape}")

    print("Loading the scaler fitted on training data (train_mlran_extractors.py)...")
    scaler = load_scaler(os.path.join(args.model_dir, "scaler.pkl"))
    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)  # transform only, never fit on test

    X_train_seq = to_model_input(X_train_scaled)
    X_test_seq = to_model_input(X_test_scaled)

    print("Loading pretrained, frozen GRU/LSTM/BiLSTM extractors...")
    gru, lstm, bilstm = load_pretrained_extractors(args.model_dir, device)

    print("Extracting fused representations (inference only, no training here)...")
    fused_train = extract_fused_features(gru, lstm, bilstm, X_train_seq, device)
    fused_test = extract_fused_features(gru, lstm, bilstm, X_test_seq, device)
    print(f"  fused_train: {fused_train.shape}, fused_test: {fused_test.shape}")

    print(
        f"\nFitting PCA on TRAINING fused representations only "
        f"(retaining {args.pca_variance:.0%} of variance)..."
    )
    pca = PCA(n_components=args.pca_variance, svd_solver="full")
    reduced_train = pca.fit_transform(fused_train)
    reduced_test = pca.transform(fused_test)  # transform only, never fit on test

    print(f"  PCA components kept        : {pca.n_components_}")
    print(f"  Variance explained         : {np.sum(pca.explained_variance_ratio_):.4f}")
    print(
        "  Note: PCA reduces dimensionality and captures the dominant "
        "variance while reducing redundancy/correlation across the fused "
        "GRU+LSTM+BiLSTM representation -- it does not remove literal "
        "duplicate rows or features."
    )

    os.makedirs(args.output_dir, exist_ok=True)

    np.save(os.path.join(args.output_dir, "X_train_pca.npy"), reduced_train)
    np.save(os.path.join(args.output_dir, "X_test_pca.npy"), reduced_test)
    np.save(os.path.join(args.output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.output_dir, "y_test.npy"), y_test)

    with open(os.path.join(args.output_dir, "pca.pkl"), "wb") as fh:
        pickle.dump(pca, fh)

    
    print(f"\nSaved PCA-reduced train/test features and labels to {args.output_dir}/")
    print("Ready for the fine-tuning/classification stage.")
    print(
        f"  X_train_pca.npy: {reduced_train.shape}   X_test_pca.npy: {reduced_test.shape}"
    )

if __name__ == "__main__":
    main()
# ==============================================================================
# ABLATION EVALUATION ADDITION
# ==============================================================================
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression


def evaluate_classifier(X_tr, y_tr, X_te, y_te):
    """Trains a downstream classifier and calculates evaluation metrics."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, y_tr)

    preds = clf.predict(X_te)
    acc = accuracy_score(y_te, preds) * 100
    f1 = f1_score(y_te, preds, average="weighted") * 100

    try:
        if len(np.unique(y_tr)) == 2:
            probs = clf.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, probs)
        else:
            probs = clf.predict_proba(X_te)
            auc = roc_auc_score(y_te, probs, multi_class="ovr")
    except Exception:
        auc = 0.0

    return acc, f1, auc


def run_ablation_evaluations(gru, lstm, bilstm, X_tr_seq, X_te_seq, y_train, y_test, device, pca_variance=0.95):
    """Runs single extractors, pair fusions, and full Tri-Rec (with/without PCA) to fill the comparison table."""
    print("\n" + "=" * 60)
    print("RUNNING ABLATION EVALUATIONS ACROSS ALL CONFIGURATIONS")
    print("=" * 60)

    # 1. Helper function for single extractor inference
    @torch.no_grad()
    def get_feats(model):
        all_tr, all_te = [], []
        batch_size = 256
        for start in range(0, len(X_tr_seq), batch_size):
            b = torch.tensor(X_tr_seq[start:start + batch_size], dtype=torch.float32).to(device)
            all_tr.append(model(b).cpu().numpy())
        for start in range(0, len(X_te_seq), batch_size):
            b = torch.tensor(X_te_seq[start:start + batch_size], dtype=torch.float32).to(device)
            all_te.append(model(b).cpu().numpy())
        return np.concatenate(all_tr, axis=0), np.concatenate(all_te, axis=0)

    # 2. Extract individual representations
    print("Extracting features from individual models...")
    lstm_tr, lstm_te = get_feats(lstm)
    gru_tr, gru_te = get_feats(gru)
    bilstm_tr, bilstm_te = get_feats(bilstm)

    # 3. Define the 8 ablation feature combinations
    configs = {
        "LSTM only": (lstm_tr, lstm_te),
        "GRU only": (gru_tr, gru_te),
        "BiLSTM only": (bilstm_tr, bilstm_te),
        "LSTM + GRU": (
            np.concatenate([lstm_tr, gru_tr], axis=1),
            np.concatenate([lstm_te, gru_te], axis=1)
        ),
        "LSTM + BiLSTM": (
            np.concatenate([lstm_tr, bilstm_tr], axis=1),
            np.concatenate([lstm_te, bilstm_te], axis=1)
        ),
        "GRU + BiLSTM": (
            np.concatenate([gru_tr, bilstm_tr], axis=1),
            np.concatenate([gru_te, bilstm_te], axis=1)
        ),
        "Tri-Rec without PCA": (
            np.concatenate([gru_tr, lstm_tr, bilstm_tr], axis=1),
            np.concatenate([gru_te, lstm_te, bilstm_te], axis=1)
        )
    }

    # Add 8th Configuration: Tri-Rec with PCA (proposed)
    tri_tr, tri_te = configs["Tri-Rec without PCA"]
    pca_eval = PCA(n_components=pca_variance, svd_solver="full")
    tri_pca_tr = pca_eval.fit_transform(tri_tr)
    tri_pca_te = pca_eval.transform(tri_te)

    configs["Tri-Rec with PCA (proposed)"] = (tri_pca_tr, tri_pca_te)

    # 4. Evaluate each configuration
    print("\nEvaluating downstream classification metrics...\n")
    results = []
    
    for name, (X_tr, X_te) in configs.items():
        acc, f1, auc = evaluate_classifier(X_tr, y_train, X_te, y_test)
        results.append((name, acc, f1, auc))

    # 5. Print formatted markdown table
    print("\n" + "| Configuration | Acc. (%) | F1 (%) | ROC-AUC |")
    print("| " + " | ".join(["---"] * 4) + " |")
    for name, acc, f1, auc in results:
        print(f"| {name:<26} | {acc:>8.2f} | {f1:>6.2f} | {auc:>7.4f} |")
    print("\n")


# Updated entry point replacing your original if __name__ == "__main__":
if __name__ == "__main__":
    # 1. Run main() pipeline to extract and save features
    main()

    # 2. Run ablation evaluations to output the comparison table
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/MLRan")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    args, _ = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    X_train_raw, y_train, X_test_raw, y_test, _ = load_raw(args.data_dir)
    scaler = load_scaler(os.path.join(args.model_dir, "scaler.pkl"))
    
    X_train_seq = to_model_input(scaler.transform(X_train_raw))
    X_test_seq = to_model_input(scaler.transform(X_test_raw))

    gru, lstm, bilstm = load_pretrained_extractors(args.model_dir, device)

    run_ablation_evaluations(
        gru, lstm, bilstm, 
        X_train_seq, X_test_seq, 
        y_train, y_test, 
        device, args.pca_variance
    )