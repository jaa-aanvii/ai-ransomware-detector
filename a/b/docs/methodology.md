# Methodology

## Why synthetic telemetry instead of real ransomware samples

Real ransomware binaries can't safely be run to generate training data
outside an isolated, air-gapped lab. Public datasets like MLRan exist,
but only as one aggregated feature row per whole sample -- no
timestamp axis, no per-process time series -- which can't support a
sequence model that needs to learn a *behavioral transition* (dwell ->
onset -> ramp) rather than a static fingerprint.

`generate_telemetry.py` instead simulates process-level telemetry
directly in the feature space the live collector will eventually
produce, with five run types:

| Run type | Purpose |
|---|---|
| `benign` | Everyday app behavior: steady low/moderate I/O, occasional legitimate bursts (save, compile). |
| `aggressive_benign` | Legitimate software that LOOKS like ransomware: backup tools, disk encryptors, media re-encoders, archivers. High write rate, high entropy (compressed/encrypted output), wide directory spread -- deliberately confusable with the positive class. |
| `bursty_aggressive_benign` | Harder confusable case: overlapping AV scan + backup + sync client, erratic spikes on every axis simultaneously. |
| `ransomware` | Dwell/recon phase that looks close to benign, then an onset into a ramping mass-encryption phase (variable ramp length and peak intensity, not a fixed signature). |
| `stealthy_ransomware` | Throttled ransomware variant with a much slower ramp and lower peak intensity, meant to stress-test whether the model relies on obvious high-intensity spikes alone. |

## Known gap: verify what was actually trained/evaluated on

`generate_dataset()` defaults `n_stealthy_ransomware_runs=0` and
`n_bursty_benign_runs=0`. Unless `test_sample_concise.xlsx` was
generated with these explicitly set above zero, the model has not
actually been trained or evaluated against its own hardest confusable
cases yet, even though the generator supports them. **Before quoting
any accuracy/F1 numbers in a report or demo, regenerate a dataset with
all five run types included and re-run `evaluate.py` against it.**

## Sequencing

- Feature rows are windowed at 500ms.
- Sequences are exactly 6 consecutive windows (3 seconds of behavior)
  from the SAME `run_id`/PID -- sequences never splice windows from two
  different processes together.
- A sequence's label is the label of its LAST window, matching what the
  live system actually needs to answer at each tick: "what is this
  process's risk right now", not "was this process ever malicious".
- Train/test splitting happens at the RUN level (`grouped_train_test_split`
  in `sequence_builder.py`), not the row level, so no windows from the
  same process leak across the split boundary and inflate metrics.

## Evaluation

`evaluate.py` reports accuracy, precision, recall, F1, and a full
confusion matrix / classification report -- not just accuracy, since a
security detector's false-negative and false-positive costs are very
different (a missed ransomware run is catastrophic; a false alarm that
suspends a legitimate backup job is disruptive but recoverable).

## Limitations to state explicitly in any writeup

- All training/eval data is synthetic; no validation against real
  ransomware samples exists or is planned (for safety reasons).
- Live PID attribution in `real_collector.py` is a best-effort
  IO-delta heuristic, not true kernel-level process attribution (see
  `architecture.md`).
- The response engine defaults to dry-run; live suspend/terminate
  actions have not been validated against a real attack, only against
  the dataset-replay path.
