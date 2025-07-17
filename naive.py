import pandas as pd

# Load cleaned data
feedback = pd.read_csv('feedback_cleaned.csv')

# Count number of 'Good' feedback for each restaurant
good_counts = feedback[feedback['Feedback'] == 'Good'].groupby('MatchedName').size().sort_values(ascending=False)

# Get the top 3
top_3 = good_counts.head(3).index.tolist()
print("Top 3 recommended restaurants:")
for i, name in enumerate(top_3, 1):
    print(f"{i}. {name}")

# Save to file
pd.DataFrame({'Top 3 Recommended Restaurants': top_3}).to_csv('top3_recommendations.csv', index=False)
print("Top 3 saved to top3_recommendations.csv")
