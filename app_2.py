from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import joblib

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

# Save the similarity matrix to a pkl file
joblib.dump(similarity_df, 'similarity_matrix.pkl')
print("Similarity matrix saved to similarity_matrix.pkl")

# Display the similarity DataFrame
print(similarity_df.head())
