import pandas as pd
import numpy as np
import re

# === USER FEATURE EXTRACTION ===
SURVEY_FILE = 'Michelin Restaurant Recommender.xlsx'
survey_df = pd.read_excel(SURVEY_FILE)
survey_df.columns = survey_df.columns.str.strip()

user_features = pd.DataFrame()
user_features['user_id'] = survey_df['Respondent ID']

# Feature 1: Wine pairing
wine_col = [col for col in survey_df.columns if 'wine pairing' in col.lower()]
if wine_col:
    wine_col = wine_col[0]
    user_features['enjoys_wine_pairing'] = survey_df[wine_col].str.lower().str.contains('yes|love|always', na=False).astype(int)
else:
    user_features['enjoys_wine_pairing'] = 0

# Feature 2: Easy reservation preference
resv_col = 'Unnamed: 96'
user_features['prefers_easy_reservation'] = survey_df[resv_col].str.contains('easier', case=False, na=False).astype(int) if resv_col in survey_df.columns else 0

# Feature 3: Ignore Michelin writeups
opinion_col = 'Are there any factors you wish were better represented in Michelin recommendations?'
user_features['ignores_michelin_reviews'] = survey_df[opinion_col].str.lower().str.contains('ignore|don\'t read|not useful', na=False).astype(int) if opinion_col in survey_df.columns else 0

# Feature 4: Keywords from memorable experience
keywords = ['service', 'ambiance', 'plating', 'wine', 'view', 'staff', 'creative', 'presentation']
mem_col = 'What is your favorite Michelin experience, and what made it memorable?'

def keyword_features(text, keyword):
    if pd.isnull(text): return 0
    return int(bool(re.search(fr'\b{keyword}\b', text.lower())))

for kw in keywords:
    user_features[f'memorable_{kw}'] = survey_df[mem_col].apply(lambda x: keyword_features(x, kw))

# Feature 5: Keywords from disappointment
dis_col = 'Have you ever been disappointed by a Michelin restaurant? Why?'
neg_keywords = ['overpriced', 'rushed', 'bland', 'small portion', 'cold', 'unfriendly']

for kw in neg_keywords:
    user_features[f'disappointed_{kw.replace(" ", "_")}'] = survey_df[dis_col].apply(lambda x: keyword_features(x, kw))

# Save user features
user_features.to_csv("user_features.csv", index=False, encoding="utf-8-sig")
print("user_features.csv exported with shape:", user_features.shape)

# === RESTAURANT FEATURE EXTRACTION ===
MICHELIN_FILE = 'Michelin_List.xlsx'
df = pd.read_excel(MICHELIN_FILE)
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

# Fix Cuisine Type column name
df['Cuisine Type'] = df['Cusine Type'] if 'Cusine Type' in df.columns else df['Cuisine Type']
cuisine_dummies = pd.get_dummies(df['Cuisine Type'], prefix='cuisine')
state_dummies = pd.get_dummies(df['State'], prefix='state')

# Combine
restaurant_features = pd.concat([df[['Restaurant Name', 'Star', 'City', 'State', 'price_tier']], cuisine_dummies, state_dummies], axis=1)

# Save restaurant features
restaurant_features.to_csv("restaurant_features.csv", index=False, encoding="utf-8-sig")
print("restaurant_features.csv exported with shape:", restaurant_features.shape)
