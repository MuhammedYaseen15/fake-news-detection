import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load cleaned data
data = pd.read_csv("Data/cleaned_news.csv")
data = data.dropna(subset=['clean_text'])

X = data['clean_text'].astype(str)
y = data['label']

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_tfidf = tfidf.fit_transform(X)

# Save vectorizer and features
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(X_tfidf, "X_tfidf.pkl")
joblib.dump(y, "labels.pkl")

print("TF-IDF extraction completed and saved!")
print("TF-IDF Shape:", X_tfidf.shape)
