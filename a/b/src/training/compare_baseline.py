import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score
)

# 1. Setup Paths & Reproducibility
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "MLRan_processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 2. Load MLRan Train/Test Data
X_train = np.load(DATA_DIR / "X_train_pca.npy")
X_test = np.load(DATA_DIR / "X_test_pca.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

def get_metrics(y_true, y_pred, y_proba, model_name, paper_ref=""):
    return {
        "Model / Framework": model_name,
        "Reference": paper_ref,
        "Accuracy (%)": round(accuracy_score(y_true, y_pred) * 100, 2),
        "Precision (%)": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "Recall (%)": round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_proba), 4),
        "PR-AUC": round(average_precision_score(y_true, y_proba), 4),
    }

results = []

# ------------------------------------------------------------------
# Setup PyTorch Tensors for Sequence & Representation Learning
# ------------------------------------------------------------------
seq_len, feat_dim = 21, 23
if X_train.shape[1] == seq_len * feat_dim:
    X_tr_seq = X_train.reshape(-1, seq_len, feat_dim)
    X_te_seq = X_test.reshape(-1, seq_len, feat_dim)
else:
    X_tr_seq = X_train.reshape(-1, 1, X_train.shape[1])
    X_te_seq = X_test.reshape(-1, 1, X_test.shape[1])
    feat_dim = X_train.shape[1]

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_tr_seq), torch.FloatTensor(y_train)), 
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_te_seq), torch.FloatTensor(y_test)), 
    batch_size=64, shuffle=False
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_eval_nn(model_cls, name, ref=""):
    model = model_cls(input_dim=feat_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(15):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device).unsqueeze(1)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    probas = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            out = model(bx)
            probas.extend(out.cpu().numpy().flatten())
            
    probas = np.array(probas)
    preds = (probas >= 0.5).astype(int)
    return get_metrics(y_test, preds, probas, name, ref)

# ------------------------------------------------------------------
# 1. DRTHIS Representation (Homayoun et al. [1]: CNN + LSTM)
# ------------------------------------------------------------------
class DRTHIS_CNN_LSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        # Permute for Conv1D: (batch, features, seq_len)
        x_conv = x.permute(0, 2, 1)
        x_conv = self.relu(self.conv1d(x_conv))
        x_conv = x_conv.permute(0, 2, 1)  # back to (batch, seq_len, 64)
        out, _ = self.lstm(x_conv)
        return self.sig(self.fc(out[:, -1, :]))

results.append(train_and_eval_nn(DRTHIS_CNN_LSTM, "DRTHIS (CNN+LSTM)", "Homayoun et al. [1]"))

# ------------------------------------------------------------------
# 2. Few-Shot / Siamese Metric Representation (Zhu et al. [2]: Dense Embedding)
# ------------------------------------------------------------------
class SiameseDenseEmbedding(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * (21 if input_dim == 23 else 1), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

results.append(train_and_eval_nn(SiameseDenseEmbedding, "Few-Shot Siamese Net", "Zhu et al. [2]"))

# ------------------------------------------------------------------
# 3. RansoGuard Representation (Cen et al. [3]: Standard RNN)
# ------------------------------------------------------------------
class RansoGuardRNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, 64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.sig(self.fc(out[:, -1, :]))

results.append(train_and_eval_nn(RansoGuardRNN, "RansoGuard (RNN)", "Cen et al. [3]"))

# ------------------------------------------------------------------
# 4. Pulse Representation (Gaber et al. [4]: Transformer Encoder)
# ------------------------------------------------------------------
class PulseTransformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(64, 1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x_emb = self.embedding(x)
        out = self.transformer(x_emb)
        return self.sig(self.fc(out.mean(dim=1)))

results.append(train_and_eval_nn(PulseTransformer, "Pulse (Transformer)", "Gaber et al. [4]"))

# ------------------------------------------------------------------
# 5. Behavioral Classifier Baseline (von der Assen et al. [5] / RF)
# ------------------------------------------------------------------
rf_base = RandomForestClassifier(n_estimators=100, random_state=SEED)
rf_base.fit(X_train, y_train)
y_proba_rf = rf_base.predict_proba(X_test)[:, 1]
y_pred_rf = (y_proba_rf >= 0.5).astype(int)
results.append(get_metrics(y_test, y_pred_rf, y_proba_rf, "Standard RF Baseline", "von der Assen et al. [5]"))

# ------------------------------------------------------------------
# Proposed Tri-Rec-MLP Model
# ------------------------------------------------------------------
model_files = list(MODELS_DIR.glob("best_model_*.pkl"))
if model_files:
    with open(model_files[0], "rb") as f:
        best_clf = pickle.load(f)
    y_pred_tri = best_clf.predict(X_test)
    y_proba_tri = best_clf.predict_proba(X_test)[:, 1]
    results.append(get_metrics(y_test, y_pred_tri, y_proba_tri, "Tri-Rec-MLP", "Proposed Work"))

# ------------------------------------------------------------------
# Summary Table Generation
# ------------------------------------------------------------------
df_summary = pd.DataFrame(results)

print("\n=============================================================================================")
print("                        LITERATURE BENCHMARK EVALUATION METRICS                              ")
print("=============================================================================================")
print(df_summary.to_string(index=False))
print("=============================================================================================")

csv_out = REPORTS_DIR / "literature_comparison_metrics.csv"
df_summary.to_csv(csv_out, index=False)
print(f"\nSaved literature benchmark metrics table to: {csv_out}")