# MLRan LSTM/GRU/BiLSTM Ensemble

A second, separate pipeline (`src/mlran_ensemble/`) built specifically
for the real MLRan dataset, alongside the existing synthetic-telemetry
pipeline. These are kept independent because the data is fundamentally
different in shape (see below) -- nothing in the original pipeline was
modified.

## Why a separate pipeline

The original `src/data_pipeline` + `src/training` code assumes genuine
time-windowed behavioral telemetry (write rate, entropy, etc. per
500ms window). MLRan's RFE-selected features are a **static, flat,
483-length binary feature vector per sample** (mostly "did this sample
call API X at all: yes/no") -- there is no time axis. Reusing the same
window engine would be incorrect, so this is a new, self-contained
module instead.

## Pipeline

```
MLRan_X_train_RFE.csv / MLRan_X_test_RFE.csv
        |
        v
data.py -- reshape flat 483-feature vector into (21, 23) pseudo-sequences
        |   (documented tradeoff: there's no guaranteed meaningful order
        |    across chunks -- see the module docstring)
        v
tune.py -- hyperparameter search per model: grid_search / random_search /
        |   optuna_search (default: Optuna/TPE -- Bayesian optimization,
        |   the most modern of the three)
        v
models.py -- LSTMEncoder, GRUEncoder, BiLSTMEncoder, each trained with
        |     its own output layer using its tuned hyperparameters
        v
ensemble.py -- for every sample:
        |        1. run all 3 trained encoders with return_embedding=True
        |           (output layer skipped)
        |        2. concatenate the 3 embeddings into one vector
        |        3. PCA on the concatenated embedding (reduces dimensions
        |           AND collapses redundant signal the 3 encoders learned
        |           in common)
        |        4. train FineTuningMLP on the PCA-reduced features
        v
   final prediction (ransomware / benign) + metrics on MLRan's official
   held-out test split
```

## Running it

```
pip install -r requirements.txt      # now includes optuna
python -m src.mlran_ensemble.run_pipeline
```

Options:
- `--tune-method {grid,random,optuna}` (default: optuna)
- `--trials N` -- trials for random/optuna search (default 25)
- `--skip-tuning` -- use fixed default hyperparameters, useful as a fast
  smoke test before committing to a full tuning run
- `--pca-variance 0.95` -- fraction of variance PCA should retain

Outputs land in `models/mlran_ensemble/`: each encoder's weights, the
fine-tuning model's weights, the fitted PCA transform, the winning
hyperparameters per model, and a metrics.json.

## Honest limitations

- **The "sequence" is synthetic.** Chunking a flat feature vector into
  (21, 23) groups gives LSTM/GRU/BiLSTM something to run over, as
  asked, but there's no guarantee neighboring feature columns are
  meaningfully related -- unlike genuine temporal data. Treat the
  recurrent architecture choice here as satisfying the assignment's
  requirement, not as an architecturally-necessary choice for this
  data (a plain MLP on the flat 483 features would be a reasonable,
  probably competitive, non-recurrent baseline to compare against).
- **PCA "removes duplicates" by removing correlated variance, not
  literal duplicate rows.** If any of the three encoders converge to
  similar representations, PCA will compress that overlap into fewer
  components -- but it isn't deduplicating exact repeated values.
- Each encoder is tuned independently on its own validation split; the
  fine-tuning model is not part of that hyperparameter search (it uses
  fixed epochs/LR). Extending Optuna to also tune the fine-tuning model
  and end-to-end fine-tune the encoders jointly would be the natural
  next step if these numbers need to improve further.
