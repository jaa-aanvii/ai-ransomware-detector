"""
data.py -- MLRan dataset loading for the LSTM/GRU/BiLSTM ensemble.

IMPORTANT / HONEST DESIGN NOTE
-------------------------------
MLRan's RFE-selected features are a STATIC per-sample feature vector
(483 binary indicators, mostly "did this sample call API X: yes/no"),
NOT a time series. There is no natural "timestep" axis in this data --
unlike the synthetic telemetry used elsewhere in this repo, which is
genuinely windowed behavior over time.

Recurrent architectures (LSTM/GRU/BiLSTM) need a sequence axis to do
anything an MLP couldn't. Since the task explicitly asks for these
three architectures on this dataset, this loader reshapes each flat
483-length feature vector into a pseudo-sequence of shape
(SEQ_LEN, INPUT_SIZE) by chunking the (arbitrarily ordered) feature
list into equal-sized groups. This is a real, if somewhat artificial,
technique used in tabular deep learning to let a sequence model attend
across chunks of the feature vector -- but it is NOT the same thing as
a genuine temporal sequence, and there is no guarantee that
feature-column order carries meaningful structure. This tradeoff is
called out here rather than left implicit.

483 features factors evenly as 21 x 23, so SEQ_LEN=21, INPUT_SIZE=23
is used by default (no padding needed, no information discarded).
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SEQ_LEN = 21
INPUT_SIZE = 23  # 21 * 23 = 483, matches MLRan's RFE feature count exactly

NON_FEATURE_COLUMNS = ["sample_id", "sample_type", "family_label", "type_label"]


class MLRanDataset:
    def __init__(self, data_dir="data/MLRan"):
        self.data_dir = data_dir
        self.feature_names = self._load_feature_names()
        self.scaler = StandardScaler()

    def _load_feature_names(self):
        path = os.path.join(self.data_dir, "RFE_selected_feature_names_dic.json")
        with open(path) as fh:
            return json.load(fh)

    def _load_split(self, filename):
        df = pd.read_csv(os.path.join(self.data_dir, filename))

        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
        n_features = len(feature_cols)

        if n_features != SEQ_LEN * INPUT_SIZE:
            raise ValueError(
                f"Expected {SEQ_LEN * INPUT_SIZE} feature columns, found "
                f"{n_features} in {filename}. Update SEQ_LEN/INPUT_SIZE in "
                f"data.py if MLRan's RFE feature count has changed."
            )

        X_flat = df[feature_cols].to_numpy(dtype=np.float32)
        y = df["sample_type"].to_numpy(dtype=np.float32)  # 0 = benign, 1 = ransomware
        sample_ids = df["sample_id"].to_numpy()

        return X_flat, y, sample_ids

    def load(self):
        """Returns (X_train_seq, y_train, X_test_seq, y_test) where each
        X is shaped (N, SEQ_LEN, INPUT_SIZE), scaled with a StandardScaler
        fit on train only and applied to both splits."""

        X_train_flat, y_train, _ = self._load_split("MLRan_X_train_RFE.csv")
        X_test_flat, y_test, _ = self._load_split("MLRan_X_test_RFE.csv")

        X_train_flat = self.scaler.fit_transform(X_train_flat)
        X_test_flat = self.scaler.transform(X_test_flat)

        X_train_seq = X_train_flat.reshape(-1, SEQ_LEN, INPUT_SIZE).astype(np.float32)
        X_test_seq = X_test_flat.reshape(-1, SEQ_LEN, INPUT_SIZE).astype(np.float32)

        return X_train_seq, y_train, X_test_seq, y_test
