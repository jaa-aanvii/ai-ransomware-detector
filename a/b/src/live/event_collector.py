import pandas as pd


class DatasetEventCollector:

    def __init__(self, dataset_path):

        self.df = pd.read_excel(
            dataset_path,
            sheet_name="sample_windows"
        )

        self.df = self.df.sort_values(
            ["run_id", "pid", "timestamp"]
        )

    def events(self):

        for _, row in self.df.iterrows():

            yield {
                "pid": int(row["pid"]),
                "run_id": int(row["run_id"]),
                "timestamp": float(row["timestamp"]),

                "features": [
                    float(row["write_rate"]),
                    float(row["rename_rate"]),
                    float(row["unique_dirs"]),
                    float(row["files_modified"]),
                    float(row["entropy"]),
                    float(row["dir_dispersion"]),
                    float(row["proc_cpu_pct"]),
                    float(row["handle_count"])
                ],

                "label": int(row["label"])
            }