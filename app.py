import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import os

# Load the CSV file into a DataFrame
df = pd.read_csv('product.csv')

# Sample a subset of the data
sample_size = 30
df = df.sample(sample_size, replace=False, random_state=42)

# Reset index
df = df.reset_index(drop=True)

# Convert text fields to lowercase
df['title'] = df['title'].str.lower()

# Combine all text fields into a new column
df['combined_text'] = df[['title', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'tag6', 'tag7', 'tag8']].agg(' '.join, axis=1)

# Vectorize the combined_text column
vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(df['combined_text'])

# Calculate cosine similarity
similarities = cosine_similarity(vectorized)

# Create a new DataFrame with similarity scores
similarity_df = pd.DataFrame(similarities, columns=df['id'], index=df['id'])

# Display the similarity DataFrame
print(similarity_df.head())

# Input product with the id
input_product_id = 22

# Get the similarity scores for the input product
input_product_similarity = similarity_df.loc[input_product_id]

# Get recommendations based on similarity scores
recommendations = input_product_similarity.nlargest(11).index.tolist()
recommendations.remove(input_product_id)

# Create a DataFrame with recommended product ids
recommendations_df = pd.DataFrame({'id': recommendations})

# foreach recommendation, get the title
for index, row in recommendations_df.iterrows():
    product_id = row['id']
    product_title = df.loc[df['id'] == product_id, 'title'].values[0]
    recommendations_df.at[index, 'title'] = product_title

# Display the recommendations DataFrame with titles
print(recommendations_df)


# Create a json file 'recommandations_id.json' with the id of the input product
json_filename = f'recommandations_{input_product_id}.json'
# Convert DataFrame to a list of dictionaries
recommendations_list = recommendations_df.to_dict(orient='records')

# Write the recommendations to the JSON file
try:
    with open(json_filename, 'x') as json_file:
        json.dump(recommendations_list, json_file, indent=4)
        print(f'Recommendations saved to {json_filename}')
except FileExistsError:
    print(f'Error: The file {json_filename} already exists. Please choose a different filename or remove the existing file.')
