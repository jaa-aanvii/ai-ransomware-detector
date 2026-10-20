"""
run_pipeline.py -- end-to-end CLI for the MLRan LSTM/GRU/BiLSTM ensemble.

    python -m src.mlran_ensemble.run_pipeline
    python -m src.mlran_ensemble.run_pipeline --tune-method random --trials 20
    python -m src.mlran_ensemble.run_pipeline --tune-method grid
    python -m src.mlran_ensemble.run_pipeline --skip-tuning   # use defaults, fast smoke test

Steps run:
    1. Load MLRan RFE data, reshape to pseudo-sequences (see data.py).
    2. Tune LSTM, GRU, and BiLSTM independently (Optuna by default).
    3. Train each with its best hyperparameters.
    4. Strip output layers, fuse embeddings, apply PCA.
    5. Train the fine-tuning classifier on the fused+reduced features.
    6. Report test-set metrics and save artifacts to models/mlran_ensemble/.
"""

import argparse
import json
import os
import pickle
import time

import torch

from src.mlran_ensemble.data import MLRanDataset
from src.mlran_ensemble.ensemble import DEVICE, run_ensemble_pipeline
from src.mlran_ensemble.tune import TUNING_METHODS

DEFAULT_PARAMS = {
    "hidden_dim": 64,
    "num_layers": 1,
    "dropout": 0.2,
    "learning_rate": 1e-3,
}

OUTPUT_DIR = "models/mlran_ensemble"


def tune_all_models(X_train, y_train, method, trials, epochs, device):
    search_fn = TUNING_METHODS[method]
    tuned = {}

    for name in ("lstm", "gru", "bilstm"):
        print(f"  Tuning {name.upper()} with '{method}' search...")
        start = time.time()

        if method == "grid":
            best_params, best_score = search_fn(name, X_train, y_train, epochs=epochs, device=device)
        else:
            best_params, best_score = search_fn(
                name, X_train, y_train, n_trials=trials, epochs=epochs, device=device
            )

        elapsed = time.time() - start
        print(f"    best val ROC-AUC = {best_score:.4f}  ({elapsed:.1f}s)  params = {best_params}")
        tuned[name] = best_params

    return tuned


def main():
    parser = argparse.ArgumentParser(description="MLRan LSTM/GRU/BiLSTM ensemble pipeline")
    parser.add_argument("--data-dir", default="data/MLRan")
    parser.add_argument(
        "--tune-method",
        choices=["grid", "random", "optuna"],
        default="optuna",
        help="Hyperparameter search method. Optuna (TPE/Bayesian) is the default -- "
        "the most modern of the three -- grid and random are available for comparison.",
    )
    parser.add_argument("--trials", type=int, default=25, help="Trials for random/optuna search")
    parser.add_argument("--tune-epochs", type=int, default=8, help="Epochs per tuning trial (kept short)")
    parser.add_argument("--encoder-epochs", type=int, default=30, help="Epochs for final encoder training")
    parser.add_argument("--final-epochs", type=int, default=60, help="Epochs for the fine-tuning classifier")
    parser.add_argument("--pca-variance", type=float, default=0.95, help="Variance fraction PCA should retain")
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip hyperparameter search and use fixed defaults (fast smoke test)",
    )
    args = parser.parse_args()

    device = DEVICE
    print(f"Using device: {device}\n")

    print("Loading MLRan dataset...")
    dataset = MLRanDataset(data_dir=args.data_dir)
    X_train, y_train, X_test, y_test = dataset.load()
    print(f"  train: {X_train.shape}, test: {X_test.shape}")
    print(f"  train positive rate: {y_train.mean():.3f}, test positive rate: {y_test.mean():.3f}\n")

    if args.skip_tuning:
        print("Skipping tuning, using fixed default hyperparameters for all three models.\n")
        tuned_params = {name: DEFAULT_PARAMS for name in ("lstm", "gru", "bilstm")}
    else:
        print(f"Tuning hyperparameters ({args.tune_method} search)...")
        tuned_params = tune_all_models(
            X_train, y_train, args.tune_method, args.trials, args.tune_epochs, device
        )
        print()

    print("Training final encoders, fusing embeddings, running PCA, training fine-tuning model...")
    result = run_ensemble_pipeline(
        X_train,
        y_train,
        X_test,
        y_test,
        tuned_params,
        encoder_epochs=args.encoder_epochs,
        pca_variance=args.pca_variance,
        final_epochs=args.final_epochs,
        device=device,
    )

    print("\n" + "=" * 60)
    print("RESULTS (MLRan held-out test set)")
    print("=" * 60)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {key:<10}: {result.metrics[key]:.4f}")
    print(f"  fused embedding dim before PCA : {result.metrics['fused_embedding_dim_before_pca']}")
    print(f"  PCA components kept            : {result.metrics['pca_components_kept']}")
    print(f"  PCA variance explained         : {result.metrics['pca_variance_explained']:.4f}")
    print("\nConfusion matrix:")
    print(f"  {result.metrics['confusion_matrix']}")
    print("\n" + result.metrics["classification_report"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, model in result.encoders.items():
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{name}_encoder.pth"))

    torch.save(result.final_model.state_dict(), os.path.join(OUTPUT_DIR, "finetuning_model.pth"))

    with open(os.path.join(OUTPUT_DIR, "pca.pkl"), "wb") as fh:
        pickle.dump(result.pca, fh)

    with open(os.path.join(OUTPUT_DIR, "tuned_params.json"), "w") as fh:
        json.dump(tuned_params, fh, indent=2)

    metrics_to_save = {k: v for k, v in result.metrics.items() if k != "classification_report"}
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics_to_save, fh, indent=2)

    print(f"\nSaved encoders, fine-tuning model, PCA, tuned params, and metrics to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
