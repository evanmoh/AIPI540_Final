import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score


# load data
df1 = pd.read_csv("cleaned_colab_feedback.csv")
df2 = pd.read_csv("cleaned_survey_feedback.csv")
df = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=["user_id", "MatchedName", "Label"])
user_features = pd.read_csv("user_features.csv")
restaurant_features = pd.read_csv("restaurant_features.csv")

restaurant_features["MatchedName"] = restaurant_features["Restaurant Name"]

# map IDs
user_map = {u: i for i, u in enumerate(df["user_id"].unique())}
rest_map = {r: i for i, r in enumerate(df["MatchedName"].unique())}
df["user_idx"] = df["user_id"].map(user_map)
df["rest_idx"] = df["MatchedName"].map(rest_map)
user_features["user_idx"] = user_features["user_id"].map(user_map)
restaurant_features["rest_idx"] = restaurant_features["MatchedName"].map(rest_map)

# merge features
df = df.merge(user_features.drop(columns=["user_id"]), on="user_idx", how="left")
df = df.merge(restaurant_features.drop(columns=["MatchedName", "Restaurant Name"]), on="rest_idx", how="left")

# drop non-numeric values
non_numeric_cols = df.select_dtypes(include=["object"]).columns.tolist()
X = df.drop(columns=["user_id", "MatchedName", "Label"] + non_numeric_cols)
y = df["Label"].values.astype(np.float32)  # ensure float 0/1

# Impute missing values before scaling!
imputer = SimpleImputer(strategy="most_frequent")
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Any NaN in X?", np.isnan(X).any())
print("Any Inf in X?", np.isinf(X).any())

# scaling feature
# Impute missing values before scaling.
imputer = SimpleImputer(strategy="most_frequent")  
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)


# Fill all missing values
imputer = SimpleImputer(strategy="most_frequent")

X = imputer.fit_transform(X)
# Split
X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
    X, y, df[["user_idx", "rest_idx"]].values, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X_temp, y_temp, idx_temp, test_size=0.25, random_state=42, stratify=y_temp)

# Neural net model 
class RestaurantRecNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RestaurantRecNet(X.shape[1]).to(device)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def to_tensor(x, y):
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1,1)

X_train_t, y_train_t = to_tensor(X_train, y_train)
X_val_t, y_val_t = to_tensor(X_val, y_val)
X_test_t, y_test_t = to_tensor(X_test, y_test)

print("Target min/max:", y_train_t.min().item(), y_train_t.max().item())

# training loop
n_epochs = 50
best_val_loss = float("inf")
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_t.to(device))
    loss = loss_fn(pred, y_train_t.to(device))
    loss.backward()
    optimizer.step()
    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t.to(device))
        val_loss = loss_fn(val_pred, y_val_t.to(device)).item()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_nn.pt")
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

model.load_state_dict(torch.load("best_nn.pt"))
model.eval()

# test evaluation

with torch.no_grad():
    test_proba = model(X_test_t.to(device)).cpu().numpy().flatten()
    test_pred = (test_proba > 0.5).astype(int)
print("\nTest Classification Report:")
print(classification_report(y_test, test_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, test_proba))

# ranking metrics
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
    if len(relevant) == 0: continue
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
