import os
import pickle
import warnings
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

N_TRIALS = 30  # same budget for every model
SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)  # identical folds for all models

# ------------------------------------------------------------------
# 1. Load PCA features and labels
from pathlib import Path

# Find project root (2 levels up from src/training/training_final.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "MLRan_processed"

# 1. Load PCA features and target labels
X_train = np.load(DATA_DIR / "X_train_pca.npy")
X_test = np.load(DATA_DIR / "X_test_pca.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_test = np.load(DATA_DIR / "y_test.npy")


# ------------------------------------------------------------------
# 2. Optuna objectives (5-fold CV, macro-F1, TRAIN data only)
# ------------------------------------------------------------------
def objective_mlp(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = [trial.suggest_int(f"n_units_l{i}", 32, 256, log=True) for i in range(n_layers)]
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)

    clf = MLPClassifier(
        hidden_layer_sizes=tuple(layers),
        activation=activation,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        solver="adam",
        early_stopping=True,
        random_state=SEED,
        max_iter=300,
    )
    return cross_val_score(clf, X_train, y_train, cv=CV, scoring="f1_macro", n_jobs=-1).mean()


def objective_rf(trial):
    clf = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 600, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        random_state=SEED,
        n_jobs=-1,
    )
    return cross_val_score(clf, X_train, y_train, cv=CV, scoring="f1_macro", n_jobs=-1).mean()


def objective_lr(trial):
    clf = LogisticRegression(
        C=trial.suggest_float("C", 1e-4, 1e2, log=True),
        penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        solver="saga",  # supports both l1 and l2
        max_iter=2000,
        random_state=SEED,
    )
    return cross_val_score(clf, X_train, y_train, cv=CV, scoring="f1_macro", n_jobs=-1).mean()


def run_study(name, objective_fn, n_trials=N_TRIALS):
    print(f"\n--- Tuning {name} ({n_trials} trials) ---")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
    print(f"Best CV f1_macro for {name}: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    return study.best_params, study.best_value


# ------------------------------------------------------------------
# 3. Model builders (same settings as the objectives)
# ------------------------------------------------------------------
def build_mlp(params):
    layer_sizes = tuple(params[f"n_units_l{i}"] for i in range(params["n_layers"]))
    return MLPClassifier(
        hidden_layer_sizes=layer_sizes,
        activation=params["activation"],
        alpha=params["alpha"],
        learning_rate_init=params["learning_rate_init"],
        solver="adam",
        early_stopping=True,
        random_state=SEED,
        max_iter=300,
    )


def build_rf(params):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight=params["class_weight"],
        random_state=SEED,
        n_jobs=-1,
    )


def build_lr(params):
    return LogisticRegression(
        C=params["C"],
        penalty=params["penalty"],
        class_weight=params["class_weight"],
        solver="saga",
        max_iter=2000,
        random_state=SEED,
    )


# ------------------------------------------------------------------
# 4. Pipeline
# ------------------------------------------------------------------
if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # STAGE 1: tune all 3 models with 5-fold CV on TRAIN only
    best_mlp_params, mlp_cv = run_study("MLPClassifier", objective_mlp)
    best_rf_params, rf_cv = run_study("RandomForestClassifier", objective_rf)
    best_lr_params, lr_cv = run_study("LogisticRegression", objective_lr)

    candidates = {
        "Logistic Regression (Optuna)": (build_lr, best_lr_params, lr_cv),
        "Random Forest (Optuna)":       (build_rf, best_rf_params, rf_cv),
        "MLP (Optuna)":                 (build_mlp, best_mlp_params, mlp_cv),
    }

    # STAGE 2: select winner by CV macro-F1 (no test data touched)
    print("\n=========================================================================")
    print("              MODEL SELECTION (5-fold CV macro-F1, train only)           ")
    print("=========================================================================")
    for n, (_, _, s) in candidates.items():
        print(f"{n:<32} {s:.4f}")

    best_name = max(candidates, key=lambda n: candidates[n][2])
    builder, best_params, best_cv = candidates[best_name]
    print(f"\nWinner by CV macro-F1: {best_name} ({best_cv:.4f})")

    # STAGE 3: retrain ONLY the winner on ALL training data
    best_clf = builder(best_params)
    best_clf.fit(X_train, y_train)

    # STAGE 4: ONE final test evaluation
    y_pred = best_clf.predict(X_test)
    y_proba = best_clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=========================================================================")
    print("                 FINAL TEST RESULTS (single evaluation)                  ")
    print("=========================================================================")
    print(f"Model     : {best_name}")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print("                Pred Benign   Pred Ransomware")
    print(f"True Benign     {cm[0][0]:>10d}   {cm[0][1]:>15d}")
    print(f"True Ransomware {cm[1][0]:>10d}   {cm[1][1]:>15d}")

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, target_names=["benign", "ransomware"], zero_division=0
    ))

    # Save the final model
    os.makedirs("models", exist_ok=True)
    safe_name = best_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    save_path = f"models/best_model_{safe_name}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(best_clf, f)
    print(f"\nSaved best model to {save_path}")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
    # Create figures directory
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ------------------------------------------------------------------
    # 1. Standalone Confusion Matrix
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Benign", "Ransomware"],
        yticklabels=["Benign", "Ransomware"],
    )
    plt.title(f"Confusion Matrix ({best_name})", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(fig_dir / f"confusion_matrix_{safe_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ------------------------------------------------------------------
    # 2. Combined ROC & Precision-Recall Curve (Single Figure)
    # ------------------------------------------------------------------
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color="#1f77b4", lw=2)
    plt.plot(recall, precision, label=f"PR Curve (AP = {pr_auc:.4f})", color="#ff7f0e", lw=2)

    # Baseline chance level reference
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance Level")

    plt.xlabel("False Positive Rate / Recall")
    plt.ylabel("True Positive Rate / Precision")
    plt.title(f"ROC & PR Curves ({best_name})", fontsize=12, fontweight="bold")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(fig_dir / f"combined_roc_pr_{safe_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved plots to {fig_dir}")