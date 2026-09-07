"""
generate_telemetry.py

Synthetic OS telemetry generator that produces the actual data shape
Person 1's LSTM needs: per-PID, chronologically ordered, 500ms-windowed
feature rows -- something MLRan cannot provide (MLRan is one aggregated
row per whole sample, with no timestamp axis or PID-level time series).

Each simulated "run" = one process's lifetime, sliced into 500ms windows.
  - BENIGN runs: steady low-noise behavior for the whole run
                 (e.g. an IDE, browser, compiler -- some bursty I/O but
                 no sustained high-entropy mass file rewriting).
  - RANSOMWARE runs: quiet for an initial "dwell" period, then transitions
                 into a ramping encryption phase (write_rate, rename_rate,
                 entropy, and dir_dispersion climb and stay elevated).

Every row carries pid, run_id, timestamp, window_index, the feature
values, and a ground-truth label for THAT WINDOW (0 before onset,
1 from onset onward for ransomware runs, always 0 for benign runs).
This makes the label a function of *current behavior*, not just
"which process is this" -- so the LSTM actually has to learn the
transition pattern instead of memorizing a static per-PID label.

Output: data/telemetry.csv
Columns: pid, run_id, timestamp, window_index, write_rate, rename_rate,
         unique_dirs, files_modified, entropy, dir_dispersion,
         proc_cpu_pct, handle_count, label
"""

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def set_seed(seed: int) -> None:
    """Re-seeds the generator with a fresh, independent random stream.
    Used to produce a genuinely new batch (different random draws, same
    realistic patterns) for testing, without touching the training data
    or duplicating this file."""
    global RNG
    RNG = np.random.default_rng(seed)

FEATURE_COLUMNS = [
    "write_rate",       # file writes per 500ms window
    "rename_rate",       # file renames per 500ms window
    "unique_dirs",       # distinct directories touched in window
    "files_modified",    # distinct files modified in window
    "entropy",            # mean Shannon entropy of written file contents (0-8)
    "dir_dispersion",    # spread metric across directory tree (0-1)
    "proc_cpu_pct",       # CPU usage percent during window
    "handle_count",       # open file handles at end of window
]
NUM_FEATURES = len(FEATURE_COLUMNS)

WINDOW_MS = 500


def _benign_run(pid: int, run_id: int, n_windows: int, start_ts: float) -> pd.DataFrame:
    """Everyday app: low/moderate steady-state I/O, occasional short bursts
    (e.g. saving a project, compiling), never sustained high entropy across
    many directories at once."""
    rows = []
    base_write = RNG.uniform(0.5, 4.0)
    base_cpu = RNG.uniform(5, 25)

    for w in range(n_windows):
        burst = 1.0
        if RNG.random() < 0.05:  # occasional legitimate burst (save/compile/zip)
            burst = RNG.uniform(3, 6)

        write_rate = max(0.0, RNG.normal(base_write * burst, 2.5))
        rename_rate = max(0.0, RNG.normal(0.2 * burst, 0.6))
        unique_dirs = max(1, int(RNG.normal(1 + 0.3 * burst, 1.5)))
        files_modified = max(0, int(RNG.normal(write_rate * 0.8, 2.0)))
        entropy = np.clip(RNG.normal(3.5, 1.8), 0, 8)          # text/code/mixed
        dir_dispersion = np.clip(RNG.normal(0.1 * burst, 0.09), 0, 1)
        proc_cpu_pct = np.clip(RNG.normal(base_cpu, 8), 0, 100)
        handle_count = max(1, int(RNG.normal(20, 12)))

        rows.append([
            pid, run_id, start_ts + w * (WINDOW_MS / 1000), w,
            write_rate, rename_rate, unique_dirs, files_modified,
            entropy, dir_dispersion, proc_cpu_pct, handle_count,
            0,  # label: always benign
        ])
    return rows


def _aggressive_benign_run(pid: int, run_id: int, n_windows: int, start_ts: float) -> pd.DataFrame:
    """Legitimate but ransomware-LOOKING software: backup tools, disk
    encryption utilities, media re-encoders, archive/zip jobs. These
    genuinely have high write rate, high entropy (compressed/encrypted
    output), and wide directory dispersion -- the exact confusable case
    that makes this a real detection problem instead of a toy one."""
    rows = []
    intensity = RNG.uniform(0.5, 1.0)  # how "hot" this particular tool runs

    for w in range(n_windows):
        write_rate = max(0.0, RNG.normal(35 * intensity, 15))
        rename_rate = max(0.0, RNG.normal(5 * intensity, 4))
        unique_dirs = max(1, int(RNG.normal(4 * intensity, 3)))
        files_modified = max(0, int(RNG.normal(write_rate * 0.7, 8)))
        entropy = np.clip(RNG.normal(6.5, 1.0), 0, 8)  # compressed/encrypted, but legit
        dir_dispersion = np.clip(RNG.normal(0.35 * intensity, 0.15), 0, 1)
        proc_cpu_pct = np.clip(RNG.normal(45, 15), 0, 100)
        handle_count = max(1, int(RNG.normal(25, 12)))

        rows.append([
            pid, run_id, start_ts + w * (WINDOW_MS / 1000), w,
            write_rate, rename_rate, unique_dirs, files_modified,
            entropy, dir_dispersion, proc_cpu_pct, handle_count,
            0,  # label: benign, despite looking aggressive
        ])
    return rows


def _ransomware_run(pid: int, run_id: int, n_windows: int, start_ts: float) -> pd.DataFrame:
    """Ransomware: dwell/recon phase looks close to benign, then a clear
    onset into a ramping mass-encryption phase with high write/rename rate,
    near-max entropy (ciphertext), and wide directory dispersion."""
    rows = []
    onset = RNG.integers(low=max(2, n_windows // 4), high=max(3, n_windows // 2))
    ramp_len = RNG.integers(2, 5)  # windows to reach full intensity

    for w in range(n_windows):
        if w < onset:
            # dwell/recon phase -- deliberately looks close to benign
            write_rate = max(0.0, RNG.normal(1.0, 1.2))
            rename_rate = max(0.0, RNG.normal(0.1, 0.3))
            unique_dirs = max(1, int(RNG.normal(1, 1.0)))
            files_modified = max(0, int(RNG.normal(1, 1.5)))
            entropy = np.clip(RNG.normal(3.0, 1.5), 0, 8)
            dir_dispersion = np.clip(RNG.normal(0.05, 0.05), 0, 1)
            proc_cpu_pct = np.clip(RNG.normal(8, 6), 0, 100)
            handle_count = max(1, int(RNG.normal(10, 6)))
            label = 0
        else:
            # slower, noisier ramp -- some ransomware variants throttle
            # themselves to evade rate-based detection, so intensity varies
            progress = min(1.0, (w - onset + 1) / ramp_len)
            peak_write = RNG.uniform(30, 70)      # variable ceiling, not fixed at 80
            write_rate = max(0.0, RNG.normal(10 + peak_write * progress, 12))
            rename_rate = max(0.0, RNG.normal(5 + 35 * progress, 9))
            unique_dirs = max(1, int(RNG.normal(2 + 10 * progress, 4)))
            files_modified = max(0, int(RNG.normal(write_rate * 0.9, 8)))
            entropy = np.clip(RNG.normal(6.8 + 0.6 * progress, 0.7), 0, 8)
            dir_dispersion = np.clip(RNG.normal(0.25 + 0.5 * progress, 0.15), 0, 1)
            proc_cpu_pct = np.clip(RNG.normal(30 + 35 * progress, 15), 0, 100)
            handle_count = max(1, int(RNG.normal(20 + 50 * progress, 15)))
            label = 1

        rows.append([
            pid, run_id, start_ts + w * (WINDOW_MS / 1000), w,
            write_rate, rename_rate, unique_dirs, files_modified,
            entropy, dir_dispersion, proc_cpu_pct, handle_count,
            label,
        ])
    return rows


def _stealthy_ransomware_run(pid: int, run_id: int, n_windows: int, start_ts: float) -> pd.DataFrame:
    """Harder edge case: a slower, lower-intensity ransomware variant that
    deliberately throttles itself to blend in more, used to stress-test
    whether the model relies on obvious high-intensity signatures alone."""
    rows = []
    onset = RNG.integers(low=max(2, n_windows // 4), high=max(3, n_windows // 2))
    ramp_len = RNG.integers(4, 9)  # much slower ramp than the standard case
    peak_write = RNG.uniform(15, 40)  # lower ceiling than the standard case

    for w in range(n_windows):
        if w < onset:
            write_rate = max(0.0, RNG.normal(1.5, 1.5))
            rename_rate = max(0.0, RNG.normal(0.2, 0.4))
            unique_dirs = max(1, int(RNG.normal(1, 1.2)))
            files_modified = max(0, int(RNG.normal(1.5, 1.8)))
            entropy = np.clip(RNG.normal(3.2, 1.7), 0, 8)
            dir_dispersion = np.clip(RNG.normal(0.06, 0.06), 0, 1)
            proc_cpu_pct = np.clip(RNG.normal(9, 7), 0, 100)
            handle_count = max(1, int(RNG.normal(11, 7)))
            label = 0
        else:
            progress = min(1.0, (w - onset + 1) / ramp_len)
            write_rate = max(0.0, RNG.normal(6 + peak_write * progress, 12))
            rename_rate = max(0.0, RNG.normal(3 + 18 * progress, 8))
            unique_dirs = max(1, int(RNG.normal(1 + 6 * progress, 3)))
            files_modified = max(0, int(RNG.normal(write_rate * 0.85, 7)))
            entropy = np.clip(RNG.normal(6.3 + 0.7 * progress, 0.9), 0, 8)
            dir_dispersion = np.clip(RNG.normal(0.15 + 0.4 * progress, 0.18), 0, 1)
            proc_cpu_pct = np.clip(RNG.normal(18 + 22 * progress, 14), 0, 100)
            handle_count = max(1, int(RNG.normal(14 + 30 * progress, 14)))
            label = 1
        rows.append([pid, run_id, start_ts + w * (WINDOW_MS / 1000), w,
                     write_rate, rename_rate, unique_dirs, files_modified,
                     entropy, dir_dispersion, proc_cpu_pct, handle_count, label])
    return rows


def _bursty_aggressive_benign_run(pid: int, run_id: int, n_windows: int, start_ts: float) -> pd.DataFrame:
    """Harder edge case: AV scan + backup + sync client overlapping --
    erratic spikes on every axis, sometimes simultaneously, the most
    confusable legitimate case for the model to get right."""
    rows = []
    for w in range(n_windows):
        spike = RNG.random() < 0.3
        mult = RNG.uniform(1.5, 3.0) if spike else 1.0
        write_rate = max(0.0, RNG.normal(30 * mult, 20))
        rename_rate = max(0.0, RNG.normal(8 * mult, 7))
        unique_dirs = max(1, int(RNG.normal(5 * mult, 4)))
        files_modified = max(0, int(RNG.normal(write_rate * 0.7, 10)))
        entropy = np.clip(RNG.normal(6.2, 1.3), 0, 8)
        dir_dispersion = np.clip(RNG.normal(0.3 * mult, 0.2), 0, 1)
        proc_cpu_pct = np.clip(RNG.normal(50, 20), 0, 100)
        handle_count = max(1, int(RNG.normal(28, 15)))
        rows.append([pid, run_id, start_ts + w * (WINDOW_MS / 1000), w,
                     write_rate, rename_rate, unique_dirs, files_modified,
                     entropy, dir_dispersion, proc_cpu_pct, handle_count, 0])
    return rows


def generate_dataset(
    n_benign_runs: int = 220,
    n_aggressive_benign_runs: int = 80,
    n_ransomware_runs: int = 180,
    min_windows: int = 20,
    max_windows: int = 60,
    n_stealthy_ransomware_runs: int = 0,
    n_bursty_benign_runs: int = 0,
) -> pd.DataFrame:
    all_rows = []
    pid = 1000
    run_id = 0
    ts_cursor = 0.0

    for _ in range(n_benign_runs):
        n_windows = int(RNG.integers(min_windows, max_windows))
        all_rows.extend(_benign_run(pid, run_id, n_windows, ts_cursor))
        pid += 1
        run_id += 1
        ts_cursor += n_windows * (WINDOW_MS / 1000) + RNG.uniform(1, 5)

    for _ in range(n_aggressive_benign_runs):
        n_windows = int(RNG.integers(min_windows, max_windows))
        all_rows.extend(_aggressive_benign_run(pid, run_id, n_windows, ts_cursor))
        pid += 1
        run_id += 1
        ts_cursor += n_windows * (WINDOW_MS / 1000) + RNG.uniform(1, 5)

    for _ in range(n_ransomware_runs):
        n_windows = int(RNG.integers(min_windows, max_windows))
        all_rows.extend(_ransomware_run(pid, run_id, n_windows, ts_cursor))
        pid += 1
        run_id += 1
        ts_cursor += n_windows * (WINDOW_MS / 1000) + RNG.uniform(1, 5)

    for _ in range(n_stealthy_ransomware_runs):
        n_windows = int(RNG.integers(min_windows, max_windows))
        all_rows.extend(_stealthy_ransomware_run(pid, run_id, n_windows, ts_cursor))
        pid += 1
        run_id += 1
        ts_cursor += n_windows * (WINDOW_MS / 1000) + RNG.uniform(1, 5)

    for _ in range(n_bursty_benign_runs):
        n_windows = int(RNG.integers(min_windows, max_windows))
        all_rows.extend(_bursty_aggressive_benign_run(pid, run_id, n_windows, ts_cursor))
        pid += 1
        run_id += 1
        ts_cursor += n_windows * (WINDOW_MS / 1000) + RNG.uniform(1, 5)

    columns = ["pid", "run_id", "timestamp", "window_index"] + FEATURE_COLUMNS + ["label"]
    df = pd.DataFrame(all_rows, columns=columns)
    return df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    df = generate_dataset()
    out_path = os.path.join(out_dir, "telemetry.csv")
    df.to_csv(out_path, index=False)

    n_runs = df["run_id"].nunique()
    n_windows = len(df)
    pos_rate = df["label"].mean()
    print(f"Generated {n_runs} runs, {n_windows} total windows -> {out_path}")
    print(f"Window-level positive (ransomware) rate: {pos_rate:.3f}")
