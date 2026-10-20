# Architecture

## Pipeline overview

```
                    +-------------------------+
   TRAINING PATH    |  generate_telemetry.py  |  synthetic runs:
                    |  (data_pipeline)         |  benign / aggressive-benign /
                    +------------+-------------+  ransomware / stealthy-ransomware
                                 |
                                 v
                    +-------------------------+
                    | sequence_builder.py      |  groups by run_id, builds
                    | (6, 8) sequences         |  (N, 6, 8) windows, splits
                    +------------+-------------+  train/test by RUN not row
                                 |
                                 v
                    +-------------------------+
                    | training/train.py        |  fits FeatureScaler,
                    | + finetune.py            |  trains RansomwareLSTM
                    +------------+-------------+
                                 |
                                 v
                    +-------------------------+
                    | models/*.pth + scaler.pkl|
                    +-------------------------+

                    +-------------------------+
   LIVE PATH        | real_collector.py        |  watchdog (file events)
                    | (src/live)                |  + psutil (process IO/CPU)
                    +------------+-------------+
                                 |  one feature row per (pid, 500ms window)
                                 v
                    +-------------------------+
                    | window_engine.py          |  same sliding-window logic
                    | (shared with training)    |  used at train time
                    +------------+-------------+
                                 |  (6, 8) sequence per pid, once buffered
                                 v
                    +-------------------------+
                    | training/inference.py     |  loads model + SAME scaler,
                    | (RansomwareInference)     |  outputs probability + label
                    +------------+-------------+
                                 |
                                 v
                    +-------------------------+
                    | response.py                |  suspend -> (repeat) -> terminate
                    | (ResponseEngine)           |  + JSONL incident log
                    +-------------------------+  + best-effort file quarantine
```

## Why this split

- **Training and live inference share `window_engine.py`'s sliding-window
  contract** (group by PID, chronological order, sequence length 6) so a
  model trained on the synthetic generator's windowing behaves
  identically against live-collected windows. If these two ever drift
  apart, the model silently sees out-of-distribution input shapes.
- **`monitor.py` (dataset replay) and `live_monitor.py` (real OS) are
  separate entry points.** The replay demo lets you show the model
  working against known-labeled data without needing root/admin
  privileges or a live target directory; `live_monitor.py` is the
  actual security-system path.
- **Detection (inference) and response (`ResponseEngine`) are decoupled.**
  The model only ever outputs a probability; deciding what to *do* about
  it (suspend, escalate to terminate, log, quarantine) is a separate,
  independently testable component with its own safety defaults
  (dry-run by default -- see `response.py` docstring).

## Known limitation: PID attribution

Standard file-system watchers report *what* changed, not *which process*
changed it. `real_collector.py` attributes each window's file activity to
whichever process had the largest write-byte delta during that window
(via `psutil`), which is a documented best-effort heuristic, not a
guarantee. True per-PID attribution needs kernel-level hooks (Linux
fanotify with `FAN_REPORT_PID`/`FAN_REPORT_FID`, an eBPF program, or on
Windows a minifilter driver / ETW provider) -- out of scope for a
portable Python collector, and the natural next upgrade if this moves
toward a real deployed agent.
