"""
tune.py -- hyperparameter search for each encoder (LSTM/GRU/BiLSTM).

Three methods are implemented, as requested:

  * grid_search()    -- exhaustive search over a fixed hyperparameter
                         grid. Simple and reproducible, but scales
                         exponentially with the number of
                         hyperparameters -- becomes wasteful fast.

  * random_search()  -- samples a fixed budget of random combinations
                         from the same search space. Usually finds a
                         near-optimal result with far fewer trials than
                         grid search (Bergstra & Bengio, 2012).

  * optuna_search()  -- the modern default. Uses Optuna's Tree-structured
                         Parzen Estimator (TPE), a Bayesian-optimization-
                         style sampler that models which hyperparameter
                         regions are promising based on prior trials,
                         instead of sampling blindly. It also supports
                         pruning (killing clearly-bad trials early),
                         which grid/random search can't do. This is the
                         method actually used by run_pipeline.py by
                         default; grid_search/random_search are kept
                         available for comparison or if Optuna isn't
                         installed.

Each search function trains short-schedule models on a held-out
validation split carved out of the training set (MLRan's own test set
is kept untouched until final evaluation) and picks the configuration
with the best validation ROC-AUC.
"""

import itertools
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.mlran_ensemble.models import MODEL_REGISTRY

BATCH_SIZE = 64

SEARCH_SPACE = {
    "hidden_dim": [32, 64, 128],
    "num_layers": [1, 2],
    "dropout": [0.1, 0.3],
    "learning_rate": [1e-3, 5e-4, 1e-4],
}


def _train_and_score(model_name, params, X_train, y_train, X_val, y_val, epochs=8, device="cpu"):
    input_size = X_train.shape[2]
    model_cls = MODEL_REGISTRY[model_name]

    model = model_cls(
        input_size=input_size,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    # Mini-batch training, not full-batch: with larger hidden sizes and
    # bidirectional/multi-layer configs, holding every sample's full
    # backprop-through-time graph in memory at once doesn't scale, no
    # matter how much RAM is available. Batching is the correct fix,
    # not a workaround.
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_probs = torch.sigmoid(val_logits).cpu().numpy().ravel()

    try:
        auc = roc_auc_score(y_val, val_probs)
    except ValueError:
        auc = 0.5  # degenerate validation split (single class) -- shouldn't happen here

    return auc


def _make_split(X, y, val_fraction=0.2, seed=42):
    return train_test_split(X, y, test_size=val_fraction, random_state=seed, stratify=y)


def grid_search(model_name, X, y, search_space=None, epochs=8, device="cpu"):
    """Exhaustive search over every combination in `search_space`."""
    search_space = search_space or SEARCH_SPACE
    X_train, X_val, y_train, y_val = _make_split(X, y)

    keys = list(search_space.keys())
    best_score, best_params = -1.0, None

    for combo in itertools.product(*(search_space[k] for k in keys)):
        params = dict(zip(keys, combo))
        score = _train_and_score(model_name, params, X_train, y_train, X_val, y_val, epochs, device)

        if score > best_score:
            best_score, best_params = score, params

    return best_params, best_score


def random_search(model_name, X, y, search_space=None, n_trials=15, epochs=8, device="cpu", seed=42):
    """Samples `n_trials` random combinations from `search_space`."""
    search_space = search_space or SEARCH_SPACE
    X_train, X_val, y_train, y_val = _make_split(X, y)

    rng = random.Random(seed)
    keys = list(search_space.keys())
    best_score, best_params = -1.0, None

    for _ in range(n_trials):
        params = {k: rng.choice(search_space[k]) for k in keys}
        score = _train_and_score(model_name, params, X_train, y_train, X_val, y_val, epochs, device)

        if score > best_score:
            best_score, best_params = score, params

    return best_params, best_score


def optuna_search(model_name, X, y, n_trials=25, epochs=8, device="cpu", seed=42):
    """TPE-based Bayesian search via Optuna -- the default, 'most modern'
    method. Each trial's hyperparameters are informed by the results of
    previous trials, and unpromising trials can be pruned early."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_train, X_val, y_train, y_val = _make_split(X, y)

    def objective(trial):
        params = {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.05, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        }
        return _train_and_score(model_name, params, X_train, y_train, X_val, y_val, epochs, device)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, study.best_value


TUNING_METHODS = {
    "grid": grid_search,
    "random": random_search,
    "optuna": optuna_search,
}
