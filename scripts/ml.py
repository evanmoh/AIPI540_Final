import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("data/output/training_hybrid_wcf.csv")
X = df.drop(columns=["user_id", "MatchedName", "Label", "user_idx", "rest_idx", "svd_pred_score"])
y = df["Label"]
# Include SVD and indices
X["svd_pred_score"] = df["svd_pred_score"]
X["user_idx"] = df["user_idx"]
X["rest_idx"] = df["rest_idx"]

# Remove any other object columns
non_numeric_cols = X.select_dtypes(include=["object"]).columns.tolist()
X = X.drop(columns=non_numeric_cols)

# Save feature columns 
feature_cols = X.drop(columns=["user_idx", "rest_idx"]).columns.tolist()
with open("models/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

# Impute & scale 
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_imputed = imputer.fit_transform(X.drop(columns=["user_idx", "rest_idx"]))
X_scaled = scaler.fit_transform(X_imputed)

# Split data (keep indices for @k metrics) 
X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
    X_scaled, y, df[["user_idx", "rest_idx"]].values, stratify=y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X_temp, y_temp, idx_temp, stratify=y_temp, test_size=0.25, random_state=42)

#Model candidates
models = {
    "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(
        scale_pos_weight=(len(y) - sum(y)) / sum(y),
        use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train & select best model -
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

# ---- Final evaluation ----
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
print(f"\nBest Model: {best_name}")
print(classification_report(y_test, y_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_proba))

# @5 Metrics 
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
for uidx in np.unique(idx_test[:, 0]):
    relevant = idx_test[(idx_test[:, 0] == uidx) & (y_test.values == 1)][:, 1]
    relevant = np.unique(relevant).tolist()
    if len(relevant) == 0:
        continue
    scores = []
    user_rows = np.where(idx_test[:, 0] == uidx)[0]
    for row in user_rows:
        scores.append((idx_test[row, 1], y_proba[row]))
    ranked = [r for r, _ in sorted(scores, key=lambda x: x[1], reverse=True)]
    precisions.append(precision_at_k(relevant, ranked, k))
    recalls.append(recall_at_k(relevant, ranked, k))
    ndcgs.append(ndcg_at_k(relevant, ranked, k))

print(f"\nPrecision@{k}: {np.mean(precisions):.4f}")
print(f"Recall@{k}: {np.mean(recalls):.4f}")
print(f"NDCG@{k}: {np.mean(ndcgs):.4f}")


# Save model artifacts for deployment
joblib.dump(best_model, "models/best_ml_model.joblib")
joblib.dump(imputer, "models/imputer.joblib")
joblib.dump(scaler, "models/scaler.joblib")
print("\nSaved: models/best_ml_model.joblib, models/imputer.joblib, models/scaler.joblib, models/feature_cols.json")
