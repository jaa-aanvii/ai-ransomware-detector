# AI Ransomware Detector

An LSTM-based ransomware detection system using process-level behavioral
telemetry and temporal sequence analysis.

## Features

The model uses 8 behavioral features:

1. write_rate
2. rename_rate
3. unique_dirs
4. files_modified
5. entropy
6. dir_dispersion
7. proc_cpu_pct
8. handle_count

## Temporal Processing

Telemetry is divided into 500 ms windows.

Six consecutive windows form one sequence:

(6, 8)

The LSTM receives batches in the format:

(B, 6, 8)

## Model

The project uses a PyTorch LSTM.

Architecture:

Input: 8 features
Sequence length: 6
Hidden dimension: 32
LSTM layers: 1
Output: ransomware risk score

## Training

Run:

python -m src.training.train

The trained model is saved to:

models/ransomware_lstm.pth

The scaler is saved to:

models/scaler.pkl

## Evaluation

Run:

python -m src.training.evaluate

## Testing

Run:

python -m tests.test_pipeline