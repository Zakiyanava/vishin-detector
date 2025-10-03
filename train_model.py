import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("Model trained and saved.")
