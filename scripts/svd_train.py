import pandas as pd
import numpy as np
import json
from sklearn.decomposition import TruncatedSVD

# 1. Load user-restaurant feedback matrix
df = pd.read_csv("data/output/training_hybrid_wcf.csv")
user_ids = df['user_id'].unique().tolist()
rest_names = df['MatchedName'].unique().tolist()

# 2. Build interaction matrix (users x restaurants)
user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
rest_name_to_idx = {name: i for i, name in enumerate(rest_names)}
interaction = np.zeros((len(user_ids), len(rest_names)))

for _, row in df.iterrows():
    ui = user_id_to_idx[row['user_id']]
    ri = rest_name_to_idx[row['MatchedName']]
    interaction[ui, ri] = row['Label']

# 3. SVD for collaborative filtering 
svd = TruncatedSVD(n_components=10, random_state=42)
user_latent = svd.fit_transform(interaction)
rest_latent = svd.components_.T  # shape: (n_restaurants, n_components)

# 4. Save latent features for scoring later
np.save("models/user_latent.npy", user_latent)
np.save("models/rest_latent.npy", rest_latent)

# 5. Save mappings
mappings = {
    "user_id_to_idx": user_id_to_idx,
    "rest_name_to_idx": rest_name_to_idx
}
with open("models/svd_mappings.json", "w") as f:
    json.dump(mappings, f)

print("SVD latent features and mappings saved.")