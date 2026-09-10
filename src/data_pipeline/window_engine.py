from collections import defaultdict, deque
import numpy as np


class SlidingWindowEngine:

    def __init__(
        self,
        sequence_length=6,
        feature_dimension=8,
        window_size_ms=500
    ):
        self.sequence_length = sequence_length
        self.feature_dimension = feature_dimension
        self.window_size_ms = window_size_ms

        self.history = defaultdict(
            lambda: deque(maxlen=sequence_length)
        )

    def add_window(
        self,
        pid,
        timestamp,
        features,
        label=None
    ):
        features = np.asarray(features, dtype=np.float32)

        if features.shape != (self.feature_dimension,):
            raise ValueError(
                f"Expected ({self.feature_dimension},), "
                f"got {features.shape}"
            )

        self.history[pid].append({
            "timestamp": timestamp,
            "features": features,
            "label": label
        })

        if len(self.history[pid]) < self.sequence_length:
            return None

        sequence = np.stack(
            [
                item["features"]
                for item in self.history[pid]
            ]
        )

        labels = [
            item["label"]
            for item in self.history[pid]
        ]

        sequence_label = labels[-1]

        return sequence, sequence_label

    def clear_pid(self, pid):
        if pid in self.history:
            del self.history[pid]

    def clear_all(self):
        self.history.clear()