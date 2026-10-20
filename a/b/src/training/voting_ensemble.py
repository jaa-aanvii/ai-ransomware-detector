"""
voting_ensemble.py -- Tri-Rec fused features -> PCA -> Voting (LR + RF + MLP)

Uses the same PCA features as training_final.py
(data/MLRan_processed/X_*_pca.npy, produced by mlran_feature_pipeline.py)
and the same Optuna-tuned LR / RF / MLP (same objectives, seed, folds).

Compares, on the held-out test set:
    * Logistic Regression (Optuna)
    * Random Forest (Optuna)
    * MLP (Optuna)
    * Hard Voting  (majority vote of the 3 models)
    * Soft Voting  (average of the 3 predicted probabilities)

Every model also gets a 5-fold CV macro-F1 on TRAIN only (same folds),
so the comparison is not decided by the test set.

Run from project root:
    python src/training/voting_ensemble.py

Outputs:
    reports/voting_ensemble_results.csv
    reports/voting_ensemble_results.json
    reports/figures/voting_confusion_matrices.png
    models/voting_soft.pkl, models/voting_hard.pkl
"""

import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training"))

# Reuse the exact data, folds, objectives and builders from training_final.py
from training_final import (  # noqa: E402
    CV, SEED, X_test, X_train, y_test, y_train,
    build_lr, build_mlp, build_rf,
    objective_lr, objective_mlp, objective_rf, run_study,
)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
for d in (REPORTS_DIR, FIG_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def evaluate(name, clf, cv_f1):
    y_pred = clf.predict(X_test)
    row = {
        "Model": name,
        "CV macro-F1": round(cv_f1, 4),
        "Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision (%)": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "Recall (%)": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": None,
        "PR-AUC": None,
    }
    # Hard voting has no probabilities -> no ROC/PR-AUC
    if hasattr(clf, "predict_proba") and getattr(clf, "voting", "soft") != "hard":
        y_proba = clf.predict_proba(X_test)[:, 1]
        row["ROC-AUC"] = round(roc_auc_score(y_test, y_proba), 4)
        row["PR-AUC"] = round(average_precision_score(y_test, y_proba), 4)
    cm = confusion_matrix(y_test, y_pred)
    row["TN"], row["FP"], row["FN"], row["TP"] = (int(v) for v in cm.ravel())
    return row, cm


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 1. Tune the three base models (identical to training_final.py)
    lr_params, lr_cv = run_study("LogisticRegression", objective_lr)
    rf_params, rf_cv = run_study("RandomForestClassifier", objective_rf)
    mlp_params, mlp_cv = run_study("MLPClassifier", objective_mlp)

    def base_estimators():
        return [
            ("lr", build_lr(lr_params)),
            ("rf", build_rf(rf_params)),
            ("mlp", build_mlp(mlp_params)),
        ]

    models = {
        "Logistic Regression (Optuna)": (build_lr(lr_params), lr_cv),
        "Random Forest (Optuna)": (build_rf(rf_params), rf_cv),
        "MLP (Optuna)": (build_mlp(mlp_params), mlp_cv),
    }

    # 2. Voting ensembles (CV on train only, same folds)
    for voting in ("hard", "soft"):
        ens = VotingClassifier(estimators=base_estimators(), voting=voting, n_jobs=-1)
        cv_f1 = cross_val_score(ens, X_train, y_train, cv=CV, scoring="f1_macro", n_jobs=-1).mean()
        print(f"Voting ({voting}) CV macro-F1: {cv_f1:.4f}")
        models[f"Voting Ensemble ({voting})"] = (ens, cv_f1)

    # 3. Fit on full train, evaluate once on test
    rows, cms = [], {}
    for name, (clf, cv_f1) in models.items():
        clf.fit(X_train, y_train)
        row, cm = evaluate(name, clf, cv_f1)
        rows.append(row)
        cms[name] = cm

    with open(MODELS_DIR / "voting_soft.pkl", "wb") as f:
        pickle.dump(models["Voting Ensemble (soft)"][0], f)
    with open(MODELS_DIR / "voting_hard.pkl", "wb") as f:
        pickle.dump(models["Voting Ensemble (hard)"][0], f)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("        INDIVIDUAL MODELS vs VOTING ENSEMBLE (Tri-Rec fused features -> PCA)")
    print("=" * 100)
    print(df.to_string(index=False))
    print(f"\nPCA features used: {X_train.shape[1]}")

    best = max(rows, key=lambda r: r["CV macro-F1"])
    print(f"Best by CV macro-F1: {best['Model']} ({best['CV macro-F1']:.4f})")

    df.to_csv(REPORTS_DIR / "voting_ensemble_results.csv", index=False)
    with open(REPORTS_DIR / "voting_ensemble_results.json", "w") as f:
        json.dump(
            {
                "pca_features": int(X_train.shape[1]),
                "seed": SEED,
                "best_params": {"lr": lr_params, "rf": rf_params, "mlp": mlp_params},
                "results": rows,
            },
            f, indent=2, default=str,
        )

    # 4. Confusion-matrix figure (optional: needs matplotlib + seaborn)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, len(cms), figsize=(4 * len(cms), 3.8))
        for ax, (name, cm) in zip(axes, cms.items()):
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["Benign", "Ransom"], yticklabels=["Benign", "Ransom"])
            ax.set_title(name.replace(" (Optuna)", ""), fontsize=10)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "voting_confusion_matrices.png", dpi=300, bbox_inches="tight")
        print(f"Saved figure to {FIG_DIR / 'voting_confusion_matrices.png'}")
    except ImportError:
        print("matplotlib/seaborn not installed -- skipped figure.")

    print(f"Saved results to {REPORTS_DIR / 'voting_ensemble_results.csv'}")
