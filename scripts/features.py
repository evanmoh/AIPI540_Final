import pandas as pd
import numpy as np
import re
import unidecode
from fuzzywuzzy import process, fuzz
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# file path
MICHELIN_MASTER = 'data/raw/Michelin_List.xlsx'
SURVEY_FILE = 'data/raw/Michelin Restaurant Recommender.xlsx'
COLAB_FILE = 'data/raw/MichelinColabFiltering.xlsx'
USER_FEATURES_FILE = 'data/output/user_features.csv'
RESTAURANT_FEATURES_FILE = 'data/output/restaurant_features.csv'
FINAL_OUTPUT = 'data/output/training_hybrid_wcf.csv'

# CLEANING & MATCHING
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

# loading masters
michelin_master_df = pd.read_excel(MICHELIN_MASTER)
michelin_master_df.columns = michelin_master_df.columns.str.strip()
michelin_master_df['Restaurant Name'] = michelin_master_df['Restaurant Name'].astype(str).str.strip()
if 'Cusine Type' in michelin_master_df.columns:
    michelin_master_df.rename(columns={'Cusine Type': 'Cuisine Type'}, inplace=True)
michelin_names_orig = michelin_master_df['Restaurant Name'].tolist()
michelin_names_clean = [unidecode.unidecode(n).lower().strip() for n in michelin_names_orig]

# user features
survey_df = pd.read_excel(SURVEY_FILE)
survey_df.columns = survey_df.columns.str.strip()

user_features = pd.DataFrame()
user_features['user_id'] = survey_df['Respondent ID']

wine_col = [col for col in survey_df.columns if 'wine pairing' in col.lower()]
if wine_col:
    wine_col = wine_col[0]
    user_features['enjoys_wine_pairing'] = survey_df[wine_col].str.lower().str.contains('yes|love|always', na=False).astype(int)
else:
    user_features['enjoys_wine_pairing'] = 0

resv_col = 'Unnamed: 96'
user_features['prefers_easy_reservation'] = survey_df[resv_col].str.contains('easier', case=False, na=False).astype(int) if resv_col in survey_df.columns else 0

opinion_col = 'Are there any factors you wish were better represented in Michelin recommendations?'
user_features['ignores_michelin_reviews'] = survey_df[opinion_col].str.lower().str.contains('ignore|don\'t read|not useful', na=False).astype(int) if opinion_col in survey_df.columns else 0

keywords = ['service', 'ambiance', 'plating', 'wine', 'view', 'staff', 'creative', 'presentation']
mem_col = 'What is your favorite Michelin experience, and what made it memorable?'
def keyword_features(text, keyword):
    if pd.isnull(text): return 0
    return int(bool(re.search(fr'\b{keyword}\b', text.lower())))

for kw in keywords:
    user_features[f'memorable_{kw}'] = survey_df[mem_col].apply(lambda x: keyword_features(x, kw))

dis_col = 'Have you ever been disappointed by a Michelin restaurant? Why?'
neg_keywords = ['overpriced', 'rushed', 'bland', 'small portion', 'cold', 'unfriendly']
for kw in neg_keywords:
    user_features[f'disappointed_{kw.replace(" ", "_")}'] = survey_df[dis_col].apply(lambda x: keyword_features(x, kw))

user_features.to_csv(USER_FEATURES_FILE, index=False, encoding="utf-8-sig")

# restaurant features
df = pd.read_excel(MICHELIN_MASTER)
df.columns = df.columns.str.strip()
df['Restaurant Name'] = df['Restaurant Name'].astype(str).str.strip()
def price_to_numeric(price):
    if pd.isnull(price): return None
    price = price.strip().lower()
    if "less than" in price: return 1
    if "$100–$200" in price: return 2
    if "$200–$400" in price: return 3
    if "$400–$600" in price: return 4
    if "$600" in price: return 5
    return 0

df['price_tier'] = df['Price Range'].apply(price_to_numeric)
df['Cuisine Type'] = df['Cuisine Type'] if 'Cuisine Type' in df.columns else df['Cusine Type']
cuisine_dummies = pd.get_dummies(df['Cuisine Type'], prefix='cuisine')
state_dummies = pd.get_dummies(df['State'], prefix='state')

restaurant_features = pd.concat([df[['Restaurant Name', 'Star', 'City', 'State', 'price_tier']], cuisine_dummies, state_dummies], axis=1)
restaurant_features.to_csv(RESTAURANT_FEATURES_FILE, index=False, encoding="utf-8-sig")

# interactions
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

survey = pd.read_excel(SURVEY_FILE)
like_col = "Which Michelin-starred restaurants anywhere in the world do you personally recommend? (List as many as you want. Please include restaurant name & city/country)"
dislike_col = "Are there any Michelin-starred restaurants you would NOT recommend or found disappointing? (List as many as you want. Please include restaurant name & city/country. Optional: explain why.)"
survey_rows = build_rows(survey, like_col, dislike_col, 'Respondent ID')

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

all_rows = survey_rows + colab_rows
df_interact = pd.DataFrame(all_rows).drop_duplicates(subset=["user_id", "MatchedName", "Label"])

# SVD
print("Building user-restaurant matrix for collaborative filtering...")
from sklearn.decomposition import TruncatedSVD

# Making sure every user and restaurant gets an index
user_idx_map = {u: i for i, u in enumerate(df_interact['user_id'].unique())}
rest_idx_map = {r: i for i, r in enumerate(df_interact['MatchedName'].unique())}
df_interact['user_idx'] = df_interact['user_id'].map(user_idx_map)
df_interact['rest_idx'] = df_interact['MatchedName'].map(rest_idx_map)

# Build user-item matrix 
n_users = len(user_idx_map)
n_rests = len(rest_idx_map)
interaction_matrix = np.zeros((n_users, n_rests))
for _, row in df_interact.iterrows():
    interaction_matrix[row['user_idx'], row['rest_idx']] = row['Label']

# SVD decomposition (matrix factorization)
svd = TruncatedSVD(n_components=20, random_state=42)
svd_user = svd.fit_transform(interaction_matrix)
svd_item = svd.components_.T

# Predicted score for every (user, restaurant)
df_interact['svd_pred_score'] = [np.dot(svd_user[int(u)], svd_item[int(r)]) for u, r in zip(df_interact['user_idx'], df_interact['rest_idx'])]

# final hybrid
# Merge in features
df_interact = df_interact.merge(user_features, on='user_id', how='left')
df_interact = df_interact.merge(restaurant_features.rename(columns={'Restaurant Name':'MatchedName'}), on='MatchedName', how='left')

df_interact.to_csv(FINAL_OUTPUT, index=False, encoding='utf-8-sig')
print("Exported:", FINAL_OUTPUT)
print("Shape:", df_interact.shape)
print("Columns:", df_interact.columns.tolist())
