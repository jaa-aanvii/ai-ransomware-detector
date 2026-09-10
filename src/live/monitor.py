import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.data_pipeline.window_engine import SlidingWindowEngine
from src.data_pipeline.scaler import FeatureScaler
from src.live.pid_tracker import PIDTracker
from src.live.event_collector import DatasetEventCollector
from src.training.inference import RansomwareInference


class RansomwareMonitor:

    def __init__(
        self,
        dataset_path,
        model_path,
        scaler_path
    ):

        self.collector = DatasetEventCollector(
            dataset_path
        )

        self.tracker = PIDTracker()

        self.window_engine = SlidingWindowEngine(
            sequence_length=6,
            feature_dimension=8,
            window_size_ms=500
        )

        self.scaler = FeatureScaler()

        self.scaler.load(
            scaler_path
        )

        self.inference = RansomwareInference(
            model_path
        )

    def run(self):

        print("Starting ransomware monitor...\n")

        for event in self.collector.events():

            pid = event["pid"]

            self.tracker.update(
                pid,
                event["timestamp"]
            )

            # Normalize the current window
            scaled_features = self.scaler.transform(
                [event["features"]]
            )[0]

            result = self.window_engine.add_window(
                pid=pid,
                timestamp=event["timestamp"],
                features=scaled_features,
                label=event["label"]
            )

            if result is None:
                continue

            sequence, actual_label = result

            prediction = self.inference.predict(
                sequence
            )

            print(
                f"PID={pid} "
                f"time={event['timestamp']:.2f}s "
                f"risk={prediction['probability']:.3f} "
                f"prediction={prediction['label']} "
                f"actual={actual_label}"
            )