"""Using textblob"""
import pandas as pd
from textblob import TextBlob

# Example Code
analysis = TextBlob(" he is regular")
analysis = analysis.correct()
print(analysis.sentiment)

"""Using VADER(Valence Aware Dictionary and sentiment Reasoner) Sentiment analyzer"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Example Code
analyzer = SentimentIntensityAnalyzer()
analysis = TextBlob("his result is poor")
analysis_corrected = analysis.correct()
vs = analyzer.polarity_scores(analysis_corrected)
print(vs)


"""Multinomial Naive Bayes"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pandas as pd
import pickle

df = pd.read_csv('./Training Set.csv')

print(df.columns)
print(df.head(10))
df = df.dropna()

print(df.columns)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

model.fit(df['meeting_remarks'], df['Sentiment'])

text = "he is improving"

NB = model.predict_proba([text])
print(NB)
#negative=0
#positive=1

# saving the model
filename = './Sentiment_model.pkl'
pickle.dump(model, open(filename, 'wb'))
