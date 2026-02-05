import streamlit as st
import joblib

# Load model & vectorizer
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# App title
st.title("📰 Fake News Detection System")
st.write("Enter a news article or social media post below:")

# Text input
news_text = st.text_area("News Content", height=200)

# Predict button
if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        news_tfidf = tfidf.transform([news_text])
        prediction = model.predict(news_tfidf)
        probability = model.predict_proba(news_tfidf)[0]

        if prediction[0] == 1:
            st.success(f"🟢 REAL News (Confidence: {probability[1]*100:.2f}%)")
        else:
            st.error(f"🔴 FAKE News (Confidence: {probability[0]*100:.2f}%)")
