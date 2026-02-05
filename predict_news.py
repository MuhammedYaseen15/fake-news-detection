import joblib

model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

print("\nEnter news text:\n")
news = input()

news_tfidf = tfidf.transform([news])
prediction = model.predict(news_tfidf)
prob = model.predict_proba(news_tfidf)[0]

if prediction[0] == 1:
    print(f"\n🟢 REAL news (confidence: {prob[1]*100:.2f}%)")
else:
    print(f"\n🔴 FAKE news (confidence: {prob[0]*100:.2f}%)")
