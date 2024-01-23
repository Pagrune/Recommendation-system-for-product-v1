from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from typing import Dict

app = FastAPI()

# Load the CSV file into a DataFrame
df = pd.read_csv('product.csv')

# Load the similarity matrix from the saved pkl file
similarity_df = joblib.load('similarity_matrix.pkl')

@app.post("/get_recommendations")
async def get_recommendations(data: Dict[str, int]):
    try:
        # Input product with the id
        input_product_id = data['input_product_id']

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

        # Convert DataFrame to a list of dictionaries
        recommendations_list = recommendations_df.to_dict(orient='records')

        return recommendations_list
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

