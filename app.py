import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# load artifact & data
restaurant_features = pd.read_csv("data/output/restaurant_features.csv")
restaurant_features["MatchedName"] = restaurant_features["Restaurant Name"]
user_features_all = pd.read_csv("data/output/user_features.csv")

with open("models/feature_cols.json", "r") as f:
    feature_cols = json.load(f)
svd = joblib.load("models/svd_model.joblib")
with open("models/svd_mappings.json", "r") as f:
    svd_mappings = json.load(f)
rest_name_to_idx = svd_mappings["rest_name_to_idx"]
rest_idx_to_name = {int(idx): name for name, idx in rest_name_to_idx.items()}

imputer = joblib.load("models/imputer.joblib")
scaler = joblib.load("models/scaler.joblib")
model = joblib.load("models/best_ml_model.joblib")

# UI
st.title("Michelin Restaurant Recommender - Evan Moh")

all_restaurants = sorted(restaurant_features["Restaurant Name"].unique())
liked = st.multiselect("Which Michelin restaurants did you LOVE?", all_restaurants)
disliked = st.multiselect("Which Michelin restaurants did you DISLIKE?", all_restaurants)

wine_pref = st.selectbox("Do you enjoy wine pairings?", ["No preference", "Yes", "No"])
state_options = ["No preference"] + sorted(restaurant_features["State"].dropna().unique())
state = st.selectbox("Preferred State", state_options)

# Price limit
price_display = {
    "Under $100": 1,
    "$100–$200": 2,
    "$200–$400": 3,
    "$400–$600": 4,
    "Over $600": 5
}
price_max_map = {
    1: 100,
    2: 200,
    3: 400,
    4: 600,
    5: 9999
}
max_budget_choice = st.selectbox(
    "What is your maximum budget per person?",
   list(price_display.keys())
)

# candidate filtering
candidates = restaurant_features[
    ~restaurant_features["Restaurant Name"].isin(liked + disliked)
].reset_index(drop=True)

# build content features
user_feature_cols = [c for c in user_features_all.columns if c != "user_id" and c in feature_cols]
user_vec = user_features_all[user_feature_cols].mean(axis=0).values.reshape(1, -1)

if "enjoys_wine_pairing" in user_feature_cols and wine_pref != "No preference":
    idx = user_feature_cols.index("enjoys_wine_pairing")
    user_vec[0, idx] = 1 if wine_pref == "Yes" else 0

#
if "price_tier" in user_feature_cols and max_budget_choice != "No preference":
    idx = user_feature_cols.index("price_tier")
    user_vec[0, idx] = float(price_display[max_budget_choice])

if st.button("Get My Top 3 Recommendations!"):
    # session interaction vector for SVD
    n_items = len(rest_name_to_idx)
    session_interactions = np.zeros(n_items)
    for r in liked:
        idx = rest_name_to_idx.get(r)
        if idx is not None:
            session_interactions[idx] = 1
    for r in disliked:
        idx = rest_name_to_idx.get(r)
        if idx is not None:
            session_interactions[idx] = 0  

    # SVD: Predict scores for each restaurant for this user
    user_svd_features = svd.transform(session_interactions.reshape(1, -1))
    svd_scores = svd.inverse_transform(user_svd_features).flatten()

    scores = []
    for _, row in candidates.iterrows():
        # State Filter
        if state != "No preference" and row["State"] != state:
            continue
        # Max price filter
        if max_budget_choice != "No preference":
            user_max_price = price_max_map[price_display[max_budget_choice]]
            rest_price = price_max_map.get(int(row["price_tier"]), 9999)
            if rest_price > user_max_price:
                continue

        rest_name = row["MatchedName"]
        restaurant_feature_cols = [c for c in restaurant_features.columns
                                   if c not in ["Restaurant Name", "MatchedName", "rest_idx"] and c in feature_cols]
        rest_vec = row[restaurant_feature_cols].astype(float).values.reshape(1, -1)

        svd_idx = rest_name_to_idx.get(rest_name, None)
        svd_pred_score = svd_scores[svd_idx] if svd_idx is not None else 0.0

        # Build hybrid feature vector
        full_vec = np.concatenate([user_vec, rest_vec, [[svd_pred_score]]], axis=1)
        feature_dict = dict(zip(user_feature_cols + restaurant_feature_cols + ["svd_pred_score"], full_vec.flatten()))
        full_df = pd.DataFrame([feature_dict])[feature_cols]
        X_input = full_df.values
        X_input = imputer.transform(X_input)
        X_input = scaler.transform(X_input)
        prob = model.predict_proba(X_input)[:, 1][0]
        scores.append((row["Restaurant Name"], prob))
    if not scores:
        st.warning("No restaurants match your filters.")
    else:
        top3 = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
        st.header("Your Top 3 Recommended Restaurants:")
        for name, prob in top3:
            st.write(f"{name} (Predicted score: {prob:.2f})")
