import pandas as pd
import numpy as np
import re
import unidecode
from fuzzywuzzy import process, fuzz

#file path
MICHELIN_MASTER = 'data/raw/Michelin_List.xlsx'
SURVEY_FILE = 'data/raw/Michelin Restaurant Recommender.xlsx'
COLAB_FILE = 'data/raw/MichelinColabFiltering.xlsx'
USER_FEATURES_FILE = 'data/output/user_features.csv'
RESTAURANT_FEATURES_FILE = 'data/output/restaurant_features.csv'

# Clean restaurant names
def clean_entry(entry):
    if pd.isnull(entry) or not isinstance(entry, str):
        return ""
    entry = re.sub(r'\(.*?\)', '', entry)
    entry = re.split(r'[-–]', entry)[0]
    entry = unidecode.unidecode(entry)
    entry = re.sub(r'[^\w\s]', '', entry)
    return entry.strip().lower()

def split_restaurant_entries(text):
    if pd.isnull(text) or not isinstance(text, str):
        return []
    text = text.replace('\n', ',').replace(' and ', ',')
    parts = re.split(r'[;,]+', text)
    return [clean_entry(p) for p in parts if p.strip()]

def match_name_robust(name, michelin_names_clean, michelin_names_orig):
    if not name: return None, 0
    match, score = process.extractOne(name, michelin_names_clean, scorer=fuzz.token_set_ratio)
    if score >= 88:
        return michelin_names_orig[michelin_names_clean.index(match)], score
    match, score = process.extractOne(name, michelin_names_clean, scorer=fuzz.partial_ratio)
    if score >= 85:
        return michelin_names_orig[michelin_names_clean.index(match)], score
    return None, score

# Load master data
michelin_master_df = pd.read_excel(MICHELIN_MASTER)
michelin_master_df.columns = michelin_master_df.columns.str.strip()
michelin_master_df['Restaurant Name'] = michelin_master_df['Restaurant Name'].astype(str).str.strip()
if 'Cusine Type' in michelin_master_df.columns:
    michelin_master_df.rename(columns={'Cusine Type': 'Cuisine Type'}, inplace=True)
michelin_names_orig = michelin_master_df['Restaurant Name'].tolist()
michelin_names_clean = [unidecode.unidecode(n).lower().strip() for n in michelin_names_orig]

# Load user & restaurant features
user_features = pd.read_csv(USER_FEATURES_FILE)
restaurant_features = pd.read_csv(RESTAURANT_FEATURES_FILE)
restaurant_features['MatchedName'] = restaurant_features['Restaurant Name'].astype(str).str.strip()

# Build training rows from survey
def build_rows(source_df, like_col, dislike_col, id_col):
    rows = []
    for idx, row in source_df.iterrows():
        user_id = row[id_col]
        # liked restaurants
        for entry in split_restaurant_entries(row.get(like_col)):
            matched_name, score = match_name_robust(entry, michelin_names_clean, michelin_names_orig)
            if matched_name:
                rows.append({'user_id': user_id, 'MatchedName': matched_name, 'Label': 1})
        # disliked restaurants
        for entry in split_restaurant_entries(row.get(dislike_col)):
            matched_name, score = match_name_robust(entry, michelin_names_clean, michelin_names_orig)
            if matched_name:
                rows.append({'user_id': user_id, 'MatchedName': matched_name, 'Label': 0})
    return rows

# survey responses
survey = pd.read_excel(SURVEY_FILE)
like_col = "Which Michelin-starred restaurants anywhere in the world do you personally recommend? (List as many as you want. Please include restaurant name & city/country)"
dislike_col = "Are there any Michelin-starred restaurants you would NOT recommend or found disappointing? (List as many as you want. Please include restaurant name & city/country. Optional: explain why.)"
survey_rows = build_rows(survey, like_col, dislike_col, 'Respondent ID')

# Colab responses (GoodRestaurant/BadRestaurant columns)
colab = pd.read_excel(COLAB_FILE)
good_cols = [col for col in colab.columns if col.startswith('GoodRestaurant')]
bad_cols = [col for col in colab.columns if col.startswith('BadRestaurant')]
colab_rows = []
for _, row in colab.iterrows():
    user_id = row['ID']
    for col in good_cols:
        for entry in split_restaurant_entries(row.get(col)):
            matched_name, score = match_name_robust(entry, michelin_names_clean, michelin_names_orig)
            if matched_name:
                colab_rows.append({'user_id': user_id, 'MatchedName': matched_name, 'Label': 1})
    for col in bad_cols:
        for entry in split_restaurant_entries(row.get(col)):
            matched_name, score = match_name_robust(entry, michelin_names_clean, michelin_names_orig)
            if matched_name:
                colab_rows.append({'user_id': user_id, 'MatchedName': matched_name, 'Label': 0})

# combine all the feedback
all_rows = survey_rows + colab_rows
df = pd.DataFrame(all_rows).drop_duplicates(subset=["user_id", "MatchedName", "Label"])

# Merge in user features
df = df.merge(user_features, on="user_id", how="left")

#Merge in restaurant features 
df = df.merge(
    restaurant_features.drop(columns=["Restaurant Name"], errors="ignore"),
    on="MatchedName",
    how="left"
)

# Adding explicit user_id/restaurant_id index columns for downstream mapping 
df["user_idx"] = df["user_id"].astype("category").cat.codes
df["rest_idx"] = df["MatchedName"].astype("category").cat.codes

# Output: final training data for hybrid filtering
df.to_csv("data/output/training_hybrid.csv", index=False, encoding='utf-8-sig')
print("Hybrid data exported: data/output/training_hybrid.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
