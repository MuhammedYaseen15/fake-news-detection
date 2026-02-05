# Step 2: Data Understanding & Cleaning

import pandas as pd
import numpy as np
import re
import nltk

# Download required NLTK resources (run once)
nltk.download('punkt')
nltk.download('punkt_tab')   
nltk.download('stopwords')
nltk.download('wordnet')


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# 1. Load Datasets
# -----------------------------
fake_df = pd.read_csv("Data/Fake.csv")
true_df = pd.read_csv("Data/True.csv")

# Add labels
fake_df['label'] = 0   # Fake news
true_df['label'] = 1   # Real news

# Combine datasets
data = pd.concat([fake_df, true_df], axis=0)

print("Combined Dataset Shape:", data.shape)
print("\nColumns:\n", data.columns)
print("\nSample Data:\n", data.head())

# -----------------------------
# 2. Keep Only Required Columns
# -----------------------------
# Most ISOT datasets have 'text' column
data = data[['text', 'label']]

# -----------------------------
# 3. Handle Missing Values
# -----------------------------
data.dropna(inplace=True)

# -----------------------------
# 4. Remove Duplicates
# -----------------------------
data.drop_duplicates(inplace=True)

# -----------------------------
# 5. Text Cleaning
# -----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)         # Remove special characters
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

data['clean_text'] = data['text'].apply(clean_text)

# -----------------------------
# 6. Save Cleaned Data
# -----------------------------
data.to_csv("Data/cleaned_news.csv", index=False)

print("\n✅ Data cleaning completed successfully!")
