"""
live_monitor.py

The actual "AI-powered OS security system" entry point: watches a real
directory, scores real process behavior, and (optionally) acts on it.

This is intentionally a SEPARATE file from monitor.py, which stays as
the dataset-replay demo (useful for showing the model working against
known-labeled data without needing a live target directory). This file
is the live path.

Usage:
    python -m src.live.live_monitor --watch /path/to/test/folder
    python -m src.live.live_monitor --watch /path/to/test/folder --enable-actions
    python -m src.live.live_monitor --watch /path/to/test/folder --enable-actions --threshold 0.9

SAFETY: without --enable-actions, the response engine only logs what it
WOULD do (dry-run). Actions that suspend/terminate real processes only
happen when you pass --enable-actions explicitly. Point --watch at a
disposable test folder, not a folder with files you care about, until
you've validated behavior on your machine.
"""

import argparse

from src.data_pipeline.window_engine import SlidingWindowEngine
from src.live.real_collector import RealTimeEventCollector
from src.live.response import ResponseEngine
from src.training.inference import RansomwareInference


FEATURE_NAMES = [
    "Write Rate",
    "Rename Rate",
    "Unique Directories",
    "Files Modified",
    "Entropy",
    "Directory Dispersion",
    "Process CPU %",
    "Handle Count",
]


def print_header(watch_path, dry_run):
    print("\n")
    print("=" * 62)
    print("      AI RANSOMWARE DETECTOR -- LIVE MODE")
    print("=" * 62)
    print(f" Watching   : {watch_path}")
    print(f" Mode       : {'DRY RUN (logging only)' if dry_run else 'ACTIONS ENABLED'}")
    print(" Model      : Fine-tuned 2-Layer LSTM")
    print(" Sequence   : 6 windows x 500 ms")
    print("=" * 62)
    print()


def print_result(pid, timestamp, sequence, prediction, response_result):
    risk = prediction["probability"]
    label = prediction["label"]

    print("-" * 62)
    print(f"PID          : {pid}")
    print(f"Time         : {timestamp:.2f}s")
    print(f"Risk Score   : {risk * 100:.2f}%")
    print(f"Prediction   : {label}")
    print()
    print("Behavioral Features (latest window):")

    for name, value in zip(FEATURE_NAMES, sequence[-1]):
        print(f"  {name:<24}: {value:.4f}")

    if label == "RANSOMWARE":
        print()
        print("\u26a0 ALERT: RANSOMWARE BEHAVIOR DETECTED")
        if response_result and response_result["action"] != "none":
            print(
                f"  -> response: {response_result['action']} "
                f"({response_result['action_result']})"
            )
    else:
        print()
        print("\u2713 Status: PROCESS APPEARS BENIGN")

    print("-" * 62)


def run(watch_path, model_path, scaler_path, threshold, dry_run, log_path):
    print_header(watch_path, dry_run)

    collector = RealTimeEventCollector(watch_path=watch_path, window_size_ms=500)
    window_engine = SlidingWindowEngine(
        sequence_length=6, feature_dimension=8, window_size_ms=500
    )
    inference = RansomwareInference(model_path=model_path, scaler_path=scaler_path)
    response_engine = ResponseEngine(
        threshold=threshold, dry_run=dry_run, log_path=log_path
    )

    print("Watching for file-system activity... (Ctrl+C to stop)\n")

    for event in collector.events():
        pid = event["pid"]

        result = window_engine.add_window(
            pid=pid,
            timestamp=event["timestamp"],
            features=event["features"],
            label=event["label"],
        )

        if result is None:
            continue

        sequence, _ = result
        prediction = inference.predict(sequence)
        response_result = response_engine.handle_prediction(pid, prediction, sequence)

        print_result(pid, event["timestamp"], sequence, prediction, response_result)


def main():
    parser = argparse.ArgumentParser(description="Live ransomware behavior monitor")
    parser.add_argument(
        "--watch", required=True, help="Directory to monitor for suspicious activity"
    )
    parser.add_argument(
        "--model",
        default="models/proper_finetuned_lstm.pth",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--scaler", default="models/proper_scaler.pkl", help="Path to fitted scaler"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Minimum RANSOMWARE probability before a response action fires",
    )
    parser.add_argument(
        "--enable-actions",
        action="store_true",
        help="Allow the response engine to actually suspend/terminate processes. "
        "Without this flag, it only logs what it would do.",
    )
    parser.add_argument(
        "--log-path", default="logs/incidents.jsonl", help="Incident log output path"
    )

    args = parser.parse_args()

    run(
        watch_path=args.watch,
        model_path=args.model,
        scaler_path=args.scaler,
        threshold=args.threshold,
        dry_run=not args.enable_actions,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()
