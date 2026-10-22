
from pathlib import Path
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_DIR = Path("a/b/data/MLRan")

X_train = pd.read_csv(DATA_DIR / "MLRan_X_train_RFE.csv")
X_test = pd.read_csv(DATA_DIR / "MLRan_X_test_RFE.csv")
labels = pd.read_csv(DATA_DIR / "MLRan_labels.csv")
train_ids = pd.read_csv(DATA_DIR / "type_train_sampleids.csv")["sample_id"]
test_ids = pd.read_csv(DATA_DIR / "type_test_sampleids.csv")["sample_id"]

if len(X_train) != len(train_ids) or len(X_test) != len(test_ids):
    raise ValueError("Feature rows do not match the sample ID files.")

if train_ids.isin(test_ids).any():
    raise ValueError("Training and testing IDs overlap.")

if labels["sample_id"].duplicated().any():
    raise ValueError("Duplicate sample IDs found in labels.")

label_map = labels.set_index("sample_id")["sample_type"]

if not train_ids.isin(label_map.index).all():
    raise ValueError("Some training IDs have no labels.")

if not test_ids.isin(label_map.index).all():
    raise ValueError("Some testing IDs have no labels.")

y_train = train_ids.map(label_map)
y_test = test_ids.map(label_map)

if y_train.nunique() != 2 or y_test.nunique() != 2:
    raise ValueError("Expected two classes in both datasets.")

X_train = X_train.drop(columns=["sample_id", "sample_type"], errors="ignore")
X_test = X_test.drop(columns=["sample_id", "sample_type"], errors="ignore")

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and testing feature columns do not match.")

X_train = X_train.apply(pd.to_numeric, errors="raise")
X_test = X_test.apply(pd.to_numeric, errors="raise")

if X_train.isin([float("inf"), -float("inf")]).any().any():
    raise ValueError("Infinite values found in training features.")

if X_test.isin([float("inf"), -float("inf")]).any().any():
    raise ValueError("Infinite values found in testing features.")

models = {
    "Gaussian Naive Bayes": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), GaussianNB()
    ),
    "Decision Tree": make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(random_state=42)
    ),
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=42)
    ),
    "KNN": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),
    "Linear SVM": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LinearSVC(random_state=42, max_iter=10000)
    )
}

results = []

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training class counts:\n", y_train.value_counts().sort_index())
print("Testing class counts:\n", y_test.value_counts().sort_index())

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    scores = None
    if hasattr(model[-1], "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model[-1], "decision_function"):
        scores = model.decision_function(X_test)

    auc = (
        roc_auc_score(y_test, scores)
        if scores is not None else float("nan")
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision_macro": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Recall_macro": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "F1_macro": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "ROC_AUC": auc
    })

    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Macro F1: {f1_score(y_test, predictions, average='macro'):.4f}")
    print(f"ROC-AUC: {auc:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("baseline_results.csv", index=False)

print("\nFinal baseline comparison:")
print(results_df.round(4).to_string(index=False))
print("\nResults saved to baseline_results.csv")
