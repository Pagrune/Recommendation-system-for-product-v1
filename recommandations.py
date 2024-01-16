import pandas as pd
import json
import joblib

# Load the CSV file into a DataFrame
df = pd.read_csv('product.csv')

# Load the similarity matrix from the saved pkl file
similarity_df = joblib.load('similarity_matrix.pkl')

# Input product with the id
input_product_id = 4

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

# Create a json file with the id of the input product
json_filename = f'recommendations_{input_product_id}.json'

# Convert DataFrame to a list of dictionaries
recommendations_list = recommendations_df.to_dict(orient='records')

# Write the recommendations to the JSON file
try:
    with open(json_filename, 'x') as json_file:
        json.dump(recommendations_list, json_file, indent=4)
        print(f'Recommendations saved to {json_filename}')
except FileExistsError:
    print(f'Error: The file {json_filename} already exists. Please choose a different filename or remove the existing file.')
