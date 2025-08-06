import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt

# Load data
df1 = pd.read_csv("data/output/cleaned_colab_feedback.csv")
df2 = pd.read_csv("data/output/cleaned_survey_feedback.csv")
df = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=["user_id", "MatchedName", "Label"])

top_k = 5  

# Compute most popular restaurants
pop_counts = df[df["Label"] == 1]["MatchedName"].value_counts()
top_restaurants = list(pop_counts.head(top_k).index)

print(f"Top-{top_k} popular restaurants:", top_restaurants)

# Ranking metrics
def precision_at_k(relevant, recommended, k):
    return len(set(recommended[:k]) & set(relevant)) / k

def recall_at_k(relevant, recommended, k):
    if len(relevant) == 0:
        return 0.0
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)

def ndcg_at_k(relevant, recommended, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

precisions, recalls, ndcgs = [], [], []
y_true_flat = []
y_pred_flat = []

users = df["user_id"].unique()
all_rests = df["MatchedName"].unique()

for user in users:
    user_likes = set(df[(df["user_id"] == user) & (df["Label"] == 1)]["MatchedName"])
    if len(user_likes) == 0:
        continue
    recommended = top_restaurants

    precisions.append(precision_at_k(user_likes, recommended, top_k))
    recalls.append(recall_at_k(user_likes, recommended, top_k))
    ndcgs.append(ndcg_at_k(user_likes, recommended, top_k))

    for rest in all_rests:
        y_true_flat.append(1 if rest in user_likes else 0)
        y_pred_flat.append(1 if rest in recommended else 0)

# Print ranking metrics
print(f"\nNaive Recommender Precision@{top_k}: {np.mean(precisions):.4f}")
print(f"Naive Recommender Recall@{top_k}: {np.mean(recalls):.4f}")
print(f"Naive Recommender NDCG@{top_k}: {np.mean(ndcgs):.4f}")

# Confusion matrix
cm = confusion_matrix(y_true_flat, y_pred_flat)
print("\nConfusion Matrix (rows: actual, columns: predicted):")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Liked", "Liked"])
disp.plot()
plt.title(f"Naive Top-{top_k} Recommender Confusion Matrix")
plt.show()

# ROC-AUC
roc_auc = roc_auc_score(y_true_flat, y_pred_flat)
print(f"\nTest ROC-AUC: {roc_auc:.4f}")
