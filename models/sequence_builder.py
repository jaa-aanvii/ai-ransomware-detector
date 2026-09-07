"""
sequence_builder.py

Turns the flat (pid, timestamp, window_index, features..., label) telemetry
table into (B, T, F) sequences for the LSTM -- correctly this time:

  - Sequences are built ONLY from consecutive windows of the SAME run_id/pid.
    A sequence never splices together windows from two different processes
    (this was the bug in the MLRan-based version: consecutive CSV rows were
    different, unrelated samples).
  - Splitting into train/test happens at the RUN level, before any
    windowing, so no windows from the same process can appear in both
    train and test (row-level random split leaks temporal neighbors
    across the split and inflates metrics).
  - The label of a sequence = the label of its LAST window (this matches
    "what is the current risk of this process right now", which is what
    the live daemon actually needs to decide at each 500ms tick).

This module is also what Person 2 should treat as the reference
implementation for the sliding-window engine -- same grouping-by-PID +
chronological-ordering contract, same T=6 default.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def build_sequences(
    df: pd.DataFrame,
    feature_columns: list,
    seq_len: int = 6,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        X: (N, seq_len, F) float32
        y: (N, 1) float32           -- label of the last window in each sequence
        groups: (N,) int            -- run_id each sequence belongs to (for grouped splitting)
    """
    X_list, y_list, group_list = [], [], []

    for run_id, group in df.groupby("run_id", sort=False):
        group = group.sort_values("window_index")
        feats = group[feature_columns].values.astype(np.float32)
        labels = group["label"].values.astype(np.float32)

        if len(feats) < seq_len:
            continue  # run too short to form even one sequence

        for start in range(0, len(feats) - seq_len + 1, stride):
            end = start + seq_len
            X_list.append(feats[start:end])
            y_list.append(labels[end - 1])
            group_list.append(run_id)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    groups = np.array(group_list)
    return X, y, groups


def grouped_train_test_split(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    test_size: float = 0.25, seed: int = 42,
):
    """Splits by run_id (group), not by row, so no PID's windows leak
    across the train/test boundary."""
    unique_runs = np.unique(groups)
    train_runs, test_runs = train_test_split(
        unique_runs, test_size=test_size, random_state=seed
    )
    train_mask = np.isin(groups, train_runs)
    test_mask = np.isin(groups, test_runs)
    return (
        X[train_mask], y[train_mask],
        X[test_mask], y[test_mask],
    )
