import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import os

# Charger le fichier CSV
df = pd.read_csv('product.csv')
df.info()

# Sélectionner un échantillon aléatoire
sample_size = 30
df = df.sample(sample_size, replace=False, random_state=42)
df = df.reset_index(drop=True)

# Mettre le texte en minuscules
df['title'] = df['title'].str.lower()

# Créer une nouvelle colonne qui combine tous les champs de texte
df['combined_text'] = df[['title', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'tag6', 'tag7', 'tag8']].agg(' '.join, axis=1)

# Utiliser CountVectorizer pour vectoriser le texte combiné
vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(df['combined_text'])

# Calculer les similarités cosinus
similarities = cosine_similarity(vectorized)
print(similarities)

# Créer un DataFrame avec les similarités
df_similarities = pd.DataFrame(similarities, columns=df['title'], index=df['title']).reset_index()
print(df_similarities.head())

# Produit d'entrée pour lequel nous voulons des recommandations
input_product = 'italian stracciatella delight'
recommendations = df_similarities.nlargest(11, input_product)[['title']].loc[df_similarities['title'] != input_product]

# Afficher les recommandations
print("Recommandations:")
print(recommendations)

# Créer un nouveau fichier 'recommendations.json' s'il n'existe pas
if not os.path.exists('recommendations.json'):
    with open('recommendations.json', 'w') as f:
        data = recommendations.to_dict('records')
        json.dump(data, f, indent=4)
else:
    # Charger les données existantes
    with open('recommendations.json', 'r') as f:
        data = json.load(f)

    # Ajouter de nouvelles données
    new_data = recommendations.to_dict('records')
    data.extend(new_data)

    # Écrire les données mises à jour dans le fichier
    with open('recommendations.json', 'w') as f:
        json.dump(data, f, indent=4)
