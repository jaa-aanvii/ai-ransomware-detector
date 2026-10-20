"""
voting_ensemble_raw_trirec.py -- Voting ensemble on Tri-Rec feature + PCA(95%) of raw MLRan features

Features per sample:
    [ PCA(StandardScaler(483 raw MLRan features), 95% variance)  |  Tri-Rec fused PCA feature ]

    * The raw-feature PCA keeps 95% of the variance (same rule used throughout the paper).
    * The Tri-Rec feature is data/MLRan_processed/X_*_pca.npy (from mlran_feature_pipeline.py).
    * Scaler and PCA are fit on TRAIN only (inside the sklearn Pipeline).

Compares on the held-out test set (and 5-fold CV macro-F1 on train only):
    Logistic Regression, Random Forest, MLP, Hard Voting, Soft Voting

Run from project root:
    python src/training/voting_ensemble_raw_trirec.py

Outputs:
    reports/voting_raw_trirec_results.csv / .json
    reports/figures/voting_raw_trirec_confusion_matrices.png
    models/voting_raw_trirec_soft.pkl, models/voting_raw_trirec_hard.pkl
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
PCA_VARIANCE = 0.95
N_RAW = 483
NON_FEATURE_COLUMNS = ["sample_id", "sample_type", "family_label", "type_label"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "MLRan"
PCA_DIR = PROJECT_ROOT / "data" / "MLRan_processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
for d in (REPORTS_DIR, FIG_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


# ------------------------------------------------------------------
# 1. Data: raw 483 features + Tri-Rec PCA feature (same row order)
# ------------------------------------------------------------------
def load():
    tr = pd.read_csv(RAW_DIR / "MLRan_X_train_RFE.csv")
    te = pd.read_csv(RAW_DIR / "MLRan_X_test_RFE.csv")
    cols = [c for c in tr.columns if c not in NON_FEATURE_COLUMNS]
    assert len(cols) == N_RAW, f"expected {N_RAW} features, got {len(cols)}"

    y_tr, y_te = tr["sample_type"].to_numpy(int), te["sample_type"].to_numpy(int)
    tri_tr = np.load(PCA_DIR / "X_train_pca.npy")
    tri_te = np.load(PCA_DIR / "X_test_pca.npy")

    # sanity check: Tri-Rec features are in the same sample order as the raw CSVs
    assert (np.load(PCA_DIR / "y_train.npy").astype(int) == y_tr).all()
    assert (np.load(PCA_DIR / "y_test.npy").astype(int) == y_te).all()

    X_tr = np.hstack([tr[cols].to_numpy(float), tri_tr])
    X_te = np.hstack([te[cols].to_numpy(float), tri_te])
    return X_tr, y_tr, X_te, y_te, tri_tr.shape[1]


# ------------------------------------------------------------------
# 2. Models
# ------------------------------------------------------------------
def preprocessor(n_tri):
    return ColumnTransformer([
        ("raw_pca", make_pipeline(StandardScaler(), PCA(n_components=PCA_VARIANCE, random_state=SEED)),
         list(range(N_RAW))),
        ("trirec", StandardScaler(), list(range(N_RAW, N_RAW + n_tri))),
    ])


def base_models():
    return [
        ("lr", LogisticRegression(C=0.3, max_iter=3000, random_state=SEED)),
        ("rf", RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=500,
                              early_stopping=True, random_state=SEED)),
    ]


def build_all(n_tri):
    models = {}
    names = {"lr": "Logistic Regression", "rf": "Random Forest", "mlp": "MLP"}
    for key, clf in base_models():
        models[names[key]] = make_pipeline(preprocessor(n_tri), clf)
    models["Voting Ensemble (hard)"] = make_pipeline(preprocessor(n_tri), VotingClassifier(base_models(), voting="hard"))
    models["Voting Ensemble (soft)"] = make_pipeline(preprocessor(n_tri), VotingClassifier(base_models(), voting="soft"))
    return models


# ------------------------------------------------------------------
# 3. Evaluation
# ------------------------------------------------------------------
def evaluate(name, model, cv_f1, X_te, y_te):
    y_pred = model.predict(X_te)
    row = {
        "Model": name,
        "CV macro-F1": round(cv_f1, 4),
        "Accuracy (%)": round(accuracy_score(y_te, y_pred) * 100, 2),
        "Precision (%)": round(precision_score(y_te, y_pred, zero_division=0) * 100, 2),
        "Recall (%)": round(recall_score(y_te, y_pred, zero_division=0) * 100, 2),
        "F1": round(f1_score(y_te, y_pred, zero_division=0), 4),
        "ROC-AUC": None,
        "PR-AUC": None,
    }
    if "hard" not in name:  # hard voting outputs labels only
        y_proba = model.predict_proba(X_te)[:, 1]
        row["ROC-AUC"] = round(roc_auc_score(y_te, y_proba), 4)
        row["PR-AUC"] = round(average_precision_score(y_te, y_proba), 4)
    cm = confusion_matrix(y_te, y_pred)
    row["TN"], row["FP"], row["FN"], row["TP"] = (int(v) for v in cm.ravel())
    return row, cm


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te, n_tri = load()
    models = build_all(n_tri)

    rows, cms = [], {}
    for name, model in models.items():
        cv_f1 = cross_val_score(model, X_tr, y_tr, cv=CV, scoring="f1_macro", n_jobs=-1).mean()
        model.fit(X_tr, y_tr)
        row, cm = evaluate(name, model, cv_f1, X_te, y_te)
        rows.append(row)
        cms[name] = cm
        print(f"{name:<24} CV={cv_f1:.4f}  test acc={row['Accuracy (%)']:.2f}%")

    n_pca = models["Voting Ensemble (soft)"][0].named_transformers_["raw_pca"][-1].n_components_

    with open(MODELS_DIR / "voting_raw_trirec_soft.pkl", "wb") as f:
        pickle.dump(models["Voting Ensemble (soft)"], f)
    with open(MODELS_DIR / "voting_raw_trirec_hard.pkl", "wb") as f:
        pickle.dump(models["Voting Ensemble (hard)"], f)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("  INDIVIDUAL vs VOTING -- [PCA 95% of raw features + Tri-Rec feature]")
    print("=" * 100)
    print(df.to_string(index=False))
    print(f"\nRaw-feature PCA components (95% variance): {n_pca}  + Tri-Rec features: {n_tri}"
          f"  = {n_pca + n_tri} total inputs")

    df.to_csv(REPORTS_DIR / "voting_raw_trirec_results.csv", index=False)
    with open(REPORTS_DIR / "voting_raw_trirec_results.json", "w") as f:
        json.dump({"seed": SEED, "pca_variance": PCA_VARIANCE, "raw_pca_components": int(n_pca),
                   "trirec_features": int(n_tri), "results": rows}, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, len(cms), figsize=(4 * len(cms), 3.8))
        for ax, (name, cm) in zip(axes, cms.items()):
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=["Benign", "Ransom"], yticklabels=["Benign", "Ransom"])
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "voting_raw_trirec_confusion_matrices.png", dpi=300, bbox_inches="tight")
        print(f"Saved figure to {FIG_DIR / 'voting_raw_trirec_confusion_matrices.png'}")
    except ImportError:
        print("matplotlib/seaborn not installed -- skipped figure.")

    print(f"Saved results to {REPORTS_DIR / 'voting_raw_trirec_results.csv'}")
