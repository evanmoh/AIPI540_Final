import pandas as pd
import re
from fuzzywuzzy import process, fuzz
import unidecode
import numpy as np

# File paths
MICHELIN_MASTER = 'Michelin_List.xlsx'
SURVEY_FILE = 'Michelin Restaurant Recommender.xlsx'
COLAB_FILE = 'MichelinColabFiltering.xlsx'

# Manual mapping for edge cases
manual_map = {
    "narissawa": "Narisawa",
    "single thread": "SingleThread",
    "french laundery": "The French Laundry",
    "thevar singapore": "Thevar",
    "cheval blanc": "Cheval Blanc",
    "core by clare smyth": "Core by Clare Smyth",
    "jont": "Jônt",
    "moments in barcelona": "Moments",
}

# Irrelevant tokens
stopwords = {
    "usa", "nyc", "la", "sf", "il", "dc", "ca", "uk", "japan", "france",
    "no", "food", "service", "experience", "reviewed", "locally", "highly", ""
}

# Clean restaurant name
def clean_entry(entry):
    if pd.isnull(entry) or not isinstance(entry, str):
        return ""
    entry = re.sub(r'\(.*?\)', '', entry)
    entry = re.split(r'[-–]', entry)[0]
    entry = unidecode.unidecode(entry)
    entry = re.sub(r'[^\w\s]', '', entry)
    return entry.strip().lower()

# Split multi-restaurant free-text into clean list
def split_restaurant_entries(text):
    if pd.isnull(text) or not isinstance(text, str):
        return []
    text = text.replace('\n', ',').replace(' and ', ',')
    parts = re.split(r'[;,]+', text)
    results = []
    for part in parts:
        clean_part = clean_entry(part)
        for word in re.split(r'  +', clean_part):
            word = word.strip()
            if word and word not in stopwords and len(word) > 2:
                results.append(word)
    return results

# Robust fuzzy match
def match_name_robust(name, michelin_names_clean, michelin_names_orig):
    if name in manual_map:
        return manual_map[name], 100, "manual"
    match, score = process.extractOne(name, michelin_names_clean, scorer=fuzz.token_set_ratio)
    if score >= 89:
        return michelin_names_orig[michelin_names_clean.index(match)], score, "token_set_ratio"
    match, score = process.extractOne(name, michelin_names_clean, scorer=fuzz.partial_ratio)
    if score >= 85:
        return michelin_names_orig[michelin_names_clean.index(match)], score, "partial_ratio"
    match, score = process.extractOne(name, michelin_names_clean, scorer=fuzz.token_sort_ratio)
    if score >= 84:
        return michelin_names_orig[michelin_names_clean.index(match)], score, "token_sort_ratio"
    return None, score, "none"

# Load Michelin U.S. list (already filtered)
michelin_master_df = pd.read_excel(MICHELIN_MASTER)
michelin_master_df.rename(columns=lambda x: x.strip(), inplace=True)
if 'Cusine Type' in michelin_master_df.columns:
    michelin_master_df.rename(columns={'Cusine Type': 'Cuisine Type'}, inplace=True)

michelin_names_orig = michelin_master_df['Restaurant Name'].astype(str).tolist()
michelin_names_clean = [unidecode.unidecode(n).lower().strip() for n in michelin_names_orig]

# === PROCESS SURVEY ===
survey = pd.read_excel(SURVEY_FILE)
col_like = "Which Michelin-starred restaurants anywhere in the world do you personally recommend? (List as many as you want. Please include restaurant name & city/country)"
col_dislike = "Are there any Michelin-starred restaurants you would NOT recommend or found disappointing? (List as many as you want. Please include restaurant name & city/country. Optional: explain why.)"

survey_rows, unmatched = [], []

for idx, row in survey.iterrows():
    user_id = row.get('Respondent ID', idx)
    for entry in split_restaurant_entries(row.get(col_like)):
        matched_name, score, method = match_name_robust(entry, michelin_names_clean, michelin_names_orig)
        if matched_name:
            survey_rows.append({
                'user_id': user_id, 'RestaurantName': entry, 'MatchedName': matched_name,
                'Label': 1, 'MatchScore': score, 'MatchMethod': method
            })
        else:
            unmatched.append({
                'user_id': user_id, 'RestaurantName': entry, 'Label': 1, 'Source': "survey", 'Score': score
            })
    for entry in split_restaurant_entries(row.get(col_dislike)):
        matched_name, score, method = match_name_robust(entry, michelin_names_clean, michelin_names_orig)
        if matched_name:
            survey_rows.append({
                'user_id': user_id, 'RestaurantName': entry, 'MatchedName': matched_name,
                'Label': 0, 'MatchScore': score, 'MatchMethod': method
            })
        else:
            unmatched.append({
                'user_id': user_id, 'RestaurantName': entry, 'Label': 0, 'Source': "survey", 'Score': score
            })

survey_df = pd.DataFrame(survey_rows)
survey_df = survey_df.merge(michelin_master_df, left_on='MatchedName', right_on='Restaurant Name', how='left')

# === PROCESS COLLAB FILE ===
colab = pd.read_excel(COLAB_FILE)
good_cols = [col for col in colab.columns if col.startswith('GoodRestaurant')]
bad_cols = [col for col in colab.columns if col.startswith('BadRestaurant')]

colab_rows = []

for _, row in colab.iterrows():
    user_id = row['ID']
    for col in good_cols:
        for entry in split_restaurant_entries(row.get(col)):
            matched_name, score, method = match_name_robust(entry, michelin_names_clean, michelin_names_orig)
            if matched_name:
                colab_rows.append({
                    'user_id': user_id, 'RestaurantName': entry, 'MatchedName': matched_name,
                    'Label': 1, 'MatchScore': score, 'MatchMethod': method
                })
            else:
                unmatched.append({
                    'user_id': user_id, 'RestaurantName': entry, 'Label': 1, 'Source': "colab", 'Score': score
                })
    for col in bad_cols:
        for entry in split_restaurant_entries(row.get(col)):
            matched_name, score, method = match_name_robust(entry, michelin_names_clean, michelin_names_orig)
            if matched_name:
                colab_rows.append({
                    'user_id': user_id, 'RestaurantName': entry, 'MatchedName': matched_name,
                    'Label': 0, 'MatchScore': score, 'MatchMethod': method
                })
            else:
                unmatched.append({
                    'user_id': user_id, 'RestaurantName': entry, 'Label': 0, 'Source': "colab", 'Score': score
                })

colab_df = pd.DataFrame(colab_rows)
colab_df = colab_df.merge(michelin_master_df, left_on='MatchedName', right_on='Restaurant Name', how='left')

# === OUTPUTS ===
survey_df.to_csv('cleaned_survey_feedback.csv', index=False, encoding='utf-8-sig')
colab_df.to_csv('cleaned_colab_feedback.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(unmatched).to_csv('unmatched_for_review.csv', index=False, encoding='utf-8-sig')

print("✔ Finished exporting:")
print("• cleaned_survey_feedback.csv")
print("• cleaned_colab_feedback.csv")
print("• unmatched_for_review.csv")
