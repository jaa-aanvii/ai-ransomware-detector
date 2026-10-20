"""
mlran_data.py -- shared MLRan loading/scaling, used by BOTH
train_mlran_extractors.py and mlran_feature_pipeline.py so the
StandardScaler is fit exactly once, on training data only, and reused
identically everywhere else. Fitting it twice in two different scripts
would risk two slightly different scalers ending up in play.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

NON_FEATURE_COLUMNS = ["sample_id", "sample_type", "family_label", "type_label"]


def load_raw(data_dir="data/MLRan"):
    """Loads MLRan's official train/test split as-is (no scaling, no
    reshaping). Returns raw feature arrays and labels."""

    train_df = pd.read_csv(os.path.join(data_dir, "MLRan_X_train_RFE.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "MLRan_X_test_RFE.csv"))

    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLUMNS]

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["sample_type"].to_numpy(dtype=np.float32)

    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["sample_type"].to_numpy(dtype=np.float32)

    return X_train, y_train, X_test, y_test, feature_cols


def fit_and_save_scaler(X_train, save_path="models/mlran/scaler.pkl"):
    """Fits a StandardScaler on TRAINING DATA ONLY and persists it.
    Call this once, from train_mlran_extractors.py. Every other script
    should load the saved scaler with load_scaler() instead of fitting
    its own -- fitting twice on data that has since changed (e.g. after
    reshuffling) would silently create two different scalers."""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    scaler = StandardScaler()
    scaler.fit(X_train)

    with open(save_path, "wb") as fh:
        pickle.dump(scaler, fh)

    return scaler


def load_scaler(path="models/mlran/scaler.pkl"):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def to_model_input(X_scaled):
    """(N, 483) -> (N, 1, 483), matching ParallelFeatureExtractor's
    expected input shape (seq_len=1, since MLRan has no time axis)."""
    return X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1]).astype(np.float32)
