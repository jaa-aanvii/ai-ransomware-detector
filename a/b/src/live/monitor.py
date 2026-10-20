import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.data_pipeline.window_engine import SlidingWindowEngine
from src.live.pid_tracker import PIDTracker
from src.live.event_collector import DatasetEventCollector
from src.training.inference import RansomwareInference


FEATURE_NAMES = [
    "Write Rate",
    "Rename Rate",
    "Unique Directories",
    "Files Modified",
    "Entropy",
    "Directory Dispersion",
    "Process CPU %",
    "Handle Count"
]


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

        self.inference = RansomwareInference(
            model_path=model_path,
            scaler_path=scaler_path
        )

    # --------------------------------------------------------
    # DISPLAY HEADER
    # --------------------------------------------------------

    def print_header(self):

        print("\n")
        print("=" * 62)
        print("           AI RANSOMWARE BEHAVIOR DETECTOR")
        print("=" * 62)
        print(" Model      : Fine-tuned 2-Layer LSTM")
        print(" Features   : 8 behavioral features")
        print(" Sequence   : 6 windows × 500 ms")
        print(" Detection  : Behavioral anomaly classification")
        print("=" * 62)
        print()

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    def print_result(
        self,
        pid,
        timestamp,
        sequence,
        prediction,
        actual_label
    ):

        risk = prediction["probability"]
        label = prediction["label"]

        print("-" * 62)

        print(
            f"PID          : {pid}"
        )

        print(
            f"Time         : {timestamp:.2f}s"
        )

        print(
            f"Risk Score   : {risk * 100:.2f}%"
        )

        print(
            f"Prediction   : {label}"
        )

        print(
            f"Actual Label : "
            f"{'RANSOMWARE' if actual_label == 1 else 'BENIGN'}"
        )

        print()

        print("Behavioral Features:")

        latest_window = sequence[-1]

        for name, value in zip(
            FEATURE_NAMES,
            latest_window
        ):

            print(
                f"  {name:<24}: {value:.4f}"
            )

        if label == "RANSOMWARE":

            print()
            print(
                "⚠ ALERT: RANSOMWARE BEHAVIOR DETECTED"
            )

        else:

            print()
            print(
                "✓ Status: PROCESS APPEARS BENIGN"
            )

        print("-" * 62)

    # --------------------------------------------------------
    # MAIN MONITOR
    # --------------------------------------------------------

    def run(self):

        self.print_header()

        print(
            "Starting behavioral telemetry replay...\n"
        )

        for event in self.collector.events():

            pid = event["pid"]

            self.tracker.update(
                pid,
                event["timestamp"]
            )

            result = self.window_engine.add_window(
                pid=pid,
                timestamp=event["timestamp"],
                features=event["features"],
                label=event["label"]
            )

            if result is None:
                continue

            sequence, actual_label = result

            prediction = self.inference.predict(
                sequence
            )

            self.print_result(
                pid=pid,
                timestamp=event["timestamp"],
                sequence=sequence,
                prediction=prediction,
                actual_label=actual_label
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    DATASET_PATH = (
        "data/test_sample_concise.xlsx"
    )

    MODEL_PATH = (
        "models/proper_finetuned_lstm.pth"
    )

    SCALER_PATH = (
        "models/proper_scaler.pkl"
    )

    monitor = RansomwareMonitor(
        dataset_path=DATASET_PATH,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH
    )

    monitor.run()