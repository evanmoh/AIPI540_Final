import pandas as pd
from fuzzywuzzy import process, fuzz

# Load files
michelin = pd.read_excel('Michelin_List.xlsx')
user_feedback = pd.read_excel('MichelinColabFiltering.xlsx')

michelin['Restaurant Name'] = michelin['Restaurant Name'].str.strip()
michelin_names = michelin['Restaurant Name'].unique()

good_cols = [col for col in user_feedback.columns if col.startswith('GoodRestaurant')]
bad_cols = [col for col in user_feedback.columns if col.startswith('BadRestaurant')]

good_long = user_feedback.melt(id_vars=['ID'], value_vars=good_cols, value_name='RestaurantName')
good_long['Feedback'] = 'Good'
bad_long = user_feedback.melt(id_vars=['ID'], value_vars=bad_cols, value_name='RestaurantName')
bad_long['Feedback'] = 'Bad'

feedback_long = pd.concat([good_long, bad_long], ignore_index=True)
feedback_long = feedback_long[['ID', 'RestaurantName', 'Feedback']]
feedback_long = feedback_long.dropna(subset=['RestaurantName'])
feedback_long['RestaurantName'] = feedback_long['RestaurantName'].str.strip()

def match_name(name):
    if name in michelin_names:
        return name
    match, score = process.extractOne(name, michelin_names, scorer=fuzz.token_sort_ratio)
    if score >= 90:
        return match
    else:
        return None

feedback_long['MatchedName'] = feedback_long['RestaurantName'].apply(match_name)
feedback_long = feedback_long.dropna(subset=['MatchedName'])
feedback_long = feedback_long[feedback_long['MatchedName'].isin(michelin_names)]

# Optional: Add features from master list (e.g. Star, City, Cuisine Type)
feedback_long = feedback_long.merge(michelin, left_on='MatchedName', right_on='Restaurant Name', how='left')

# Save cleaned file
feedback_long.to_csv('feedback_cleaned.csv', index=False)
print("Saved cleaned data as feedback_cleaned.csv")
