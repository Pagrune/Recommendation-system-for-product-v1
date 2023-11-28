from flask import Flask
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

df = pd.read_csv('product.csv')
df.info()




sample_size = 20
df = df.sample(sample_size, replace=False, random_state=42)

df = df.reset_index()
df = df.drop('index', axis=1)


df['title'] = df['title'].str.lower()

# make a new column that combines all text fields
df2 = df['title'] + df['tag1'] + df['tag2'] + df['tag3'] + df['tag4']

# print(df2.head())

vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(df2)

similarities = cosine_similarity(vectorized)
print(similarities)

df = pd.DataFrame(similarities, columns=df['title'], index=df['title']).reset_index()
print(df.head())

input_product = 'cherry garcia delight'
recommendations = pd.DataFrame(df.nlargest(11,input_product)['title'])
recommendations = recommendations[recommendations['title']!=input_product]
print(recommendations)

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
