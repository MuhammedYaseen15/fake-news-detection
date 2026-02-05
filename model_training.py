import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load cleaned data
data = pd.read_csv("Data/cleaned_news.csv")

# 🔥 FIX 1: Remove NaN values
data = data.dropna(subset=['clean_text', 'label'])

# 🔥 FIX 2: Ensure text is string
data['clean_text'] = data['clean_text'].astype(str)

# Features and labels
X = data['clean_text']
y = data['label']

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    stop_words='english'
)

X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("✅ Model trained and saved successfully")
