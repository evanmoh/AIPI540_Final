import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ---------- Load Data ----------
df1 = pd.read_csv("cleaned_colab_feedback.csv")
df2 = pd.read_csv("cleaned_survey_feedback.csv")
df = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=["user_id", "MatchedName", "Label"])
user_features = pd.read_csv("user_features.csv")
restaurant_features = pd.read_csv("restaurant_features.csv")

# ---------- Create Consistent Name Fields ----------
restaurant_features["MatchedName"] = restaurant_features["Restaurant Name"]

# ---------- Map user and restaurant IDs ----------
user_map = {u: i for i, u in enumerate(df["user_id"].unique())}
rest_map = {r: i for i, r in enumerate(df["MatchedName"].unique())}
df["user_idx"] = df["user_id"].map(user_map)
df["rest_idx"] = df["MatchedName"].map(rest_map)
user_features["user_idx"] = user_features["user_id"].map(user_map)
restaurant_features["rest_idx"] = restaurant_features["MatchedName"].map(rest_map)

# ---------- Merge features ----------
df = df.merge(user_features.drop(columns=["user_id"]), on="user_idx", how="left")
df = df.merge(restaurant_features.drop(columns=["MatchedName", "Restaurant Name"]), on="rest_idx", how="left")

# Prepare features (drop non-numeric columns safely)
drop_cols = ["user_id", "MatchedName", "Label", "Restaurant Name"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
non_numeric_cols = X.select_dtypes(include=["object"]).columns.tolist()
X = X.drop(columns=non_numeric_cols)
y = df["Label"]

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)



# ---------- Split into train, val, test ----------
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.25, random_state=42)

# ---------- Define models ----------
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# ---------- Train and select best model ----------
best_model = None
best_score = 0
best_name = ""
for name, model in models.items():
    model.fit(X_train, y_train)
    val_preds = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, val_preds)
    print(f"{name} ROC-AUC on val: {score:.3f}")
    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

# ---------- Final test set evaluation ----------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
print(f"\nBest Model: {best_name}")
print(classification_report(y_test, y_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_proba))

# ---------- Ranking Metrics ----------
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

for uid in df["user_id"].unique():
    uidx = user_map[uid]
    relevant = df[(df["user_id"] == uid) & (df["Label"] == 1)]["rest_idx"].tolist()
    if not relevant:
        continue

    candidates = restaurant_features["rest_idx"].unique()
    scores = []

    for ridx in candidates:
        user_row = user_features[user_features["user_idx"] == uidx].drop(columns=["user_idx"], errors='ignore')
        rest_row = restaurant_features[restaurant_features["rest_idx"] == ridx].drop(columns=["rest_idx", "Restaurant Name"], errors='ignore')
        if user_row.empty or rest_row.empty:
            continue

        combined = pd.concat([user_row.reset_index(drop=True), rest_row.reset_index(drop=True)], axis=1)
        # Ensure only numeric columns and matching X columns/order
        combined = combined.reindex(columns=X.columns, fill_value=0)

        row_arr = imputer.transform(combined)
        row_arr = scaler.transform(row_arr)
        prob = best_model.predict_proba(row_arr)[0][1]
        scores.append((ridx, prob))

    ranked = [r for r, _ in sorted(scores, key=lambda x: x[1], reverse=True)]
    precisions.append(precision_at_k(relevant, ranked, k))
    recalls.append(recall_at_k(relevant, ranked, k))
    ndcgs.append(ndcg_at_k(relevant, ranked, k))

print(f"\nPrecision@{k}: {np.mean(precisions):.3f}")
print(f"Recall@{k}: {np.mean(recalls):.3f}")
print(f"NDCG@{k}: {np.mean(ndcgs):.3f}")
