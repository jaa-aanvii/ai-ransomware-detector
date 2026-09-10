FEATURES = [
    "write_rate",
    "rename_rate",
    "unique_dirs",
    "files_modified",
    "entropy",
    "dir_dispersion",
    "proc_cpu_pct",
    "handle_count",
]


class FeatureSchema:
    def __init__(self):
        self.features = FEATURES.copy()

    @property
    def dimension(self):
        return len(self.features)

    def validate(self, columns):
        missing = [f for f in self.features if f not in columns]

        if missing:
            raise ValueError(f"Missing features: {missing}")

        return True

    def get_features(self):
        return self.features.copy()