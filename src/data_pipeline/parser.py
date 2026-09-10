import pandas as pd

from .feature_schema import FeatureSchema


class DatasetParser:

    def __init__(self, file_path):
        self.file_path = file_path
        self.schema = FeatureSchema()

    def load(self):
        df = pd.read_excel(
            self.file_path,
            sheet_name="sample_windows"
        )

        self.schema.validate(df.columns)

        required = [
            "pid",
            "run_id",
            "timestamp",
            "window_index",
            "label"
        ]

        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        df = df.sort_values(
            ["run_id", "pid", "timestamp"]
        ).reset_index(drop=True)

        return df

    def get_features(self, df):
        return df[self.schema.get_features()].values

    def get_labels(self, df):
        return df["label"].values

    def get_metadata(self, df):
        return df[
            [
                "pid",
                "run_id",
                "timestamp",
                "window_index"
            ]
        ]