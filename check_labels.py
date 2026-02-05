import pandas as pd

data = pd.read_csv("Data/cleaned_news.csv")
print(data['label'].value_counts())
