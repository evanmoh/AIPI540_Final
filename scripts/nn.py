import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import json

# load data
df = pd.read_csv("data/output/training_hybrid_wcf.csv")

# feature engineering
# Drop non-numeric columns, user/rest names, and label
X = df.drop(columns=["user_id", "MatchedName", "Label"])
y = df["Label"].values.astype(np.float32)

# Drop any leftover object columns for safety
non_numeric_cols = X.select_dtypes(include=["object"]).columns.tolist()
X = X.drop(columns=non_numeric_cols)

# saving features for later use 
feature_cols = X.columns.tolist()
with open("models/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

# imputing & scaling
imputer = SimpleImputer(strategy="most_frequent")
X = imputer.fit_transform(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split data
# Also keep user/rest idx for ranking later
X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
    X, y, df[["user_idx", "rest_idx"]].values, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X_temp, y_temp, idx_temp, test_size=0.25, random_state=42, stratify=y_temp)

# handle class imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)

# Neural net
class RestaurantRecNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 16)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

def to_tensor(x, y):
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)

X_train_t, y_train_t = to_tensor(X_train, y_train)
X_val_t, y_val_t = to_tensor(X_val, y_val)
X_test_t, y_test_t = to_tensor(X_test, y_test)

# Hyper parameter tuning: LR + EARLY STOPPING
lrs = [0.001, 0.0005, 0.0001]
n_epochs = 50
patience = 8
best_val_auc = 0
best_lr = None
best_model_state = None

for lr in lrs:
    print(f"\nTraining with learning rate: {lr}")
    model = RestaurantRecNet(X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_loss = float("inf")
    no_improve = 0
    val_auc_max = 0

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t.to(device))
        loss = loss_fn(pred, y_train_t.to(device))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t.to(device))
            val_loss = loss_fn(val_pred, y_val_t.to(device)).item()
            val_logits = val_pred.cpu().numpy().flatten()
            val_proba = 1 / (1 + np.exp(-val_logits))
            val_auc = roc_auc_score(y_val, val_proba)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_nn_tmp.pt")
            no_improve = 0
            val_auc_max = val_auc
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val ROC-AUC: {val_auc:.4f}")
        if no_improve > patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if val_auc_max > best_val_auc:
        best_val_auc = val_auc_max
        best_lr = lr
        best_model_state = torch.load("models/best_nn_tmp.pt")

print(f"\nBest learning rate: {best_lr}, Best val ROC-AUC: {best_val_auc:.4f}")

# load the best model
model = RestaurantRecNet(X.shape[1]).to(device)
model.load_state_dict(best_model_state)
model.eval()
torch.save(model.state_dict(), "models/best_nn.pt")
joblib.dump(imputer, "models/imputer.joblib")
joblib.dump(scaler, "models/scaler.joblib")

# evaluate on the test set.
with torch.no_grad():
    test_logits = model(X_test_t.to(device)).cpu().numpy().flatten()
    test_proba = 1 / (1 + np.exp(-test_logits))  # Sigmoid for probability
    test_pred = (test_proba > 0.5).astype(int)

print("\nTest Classification Report:")
print(classification_report(y_test, test_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, test_proba))

# Ranking metrics per user
def precision_at_k(relevant, recommended, k):
    return len(set(recommended[:k]) & set(relevant)) / k

def recall_at_k(relevant, recommended, k):
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)

def ndcg_at_k(relevant, recommended, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

k = 5
precisions, recalls, ndcgs = [], [], []
for uidx in np.unique(idx_test[:,0]):
    relevant = idx_test[(idx_test[:,0] == uidx) & (y_test == 1)][:,1]
    relevant = np.unique(relevant).tolist()
    if len(relevant) == 0:
        continue
    scores = []
    user_rows = np.where(idx_test[:,0] == uidx)[0]
    for row in user_rows:
        scores.append((idx_test[row,1], test_proba[row]))
    ranked = [r for r, _ in sorted(scores, key=lambda x: x[1], reverse=True)]
    precisions.append(precision_at_k(relevant, ranked, k))
    recalls.append(recall_at_k(relevant, ranked, k))
    ndcgs.append(ndcg_at_k(relevant, ranked, k))

print(f"\nNeural Net Precision@{k}: {np.mean(precisions):.4f}")
print(f"Neural Net Recall@{k}: {np.mean(recalls):.4f}")
print(f"Neural Net NDCG@{k}: {np.mean(ndcgs):.4f}")
