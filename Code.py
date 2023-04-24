import spacy
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textblob")

# Load the preprocessed data
train_df = pd.read_csv("train_tweet.csv")
test_df = pd.read_csv("test_tweet.csv")

# Define a function to extract aspects from text
def extract_aspects(text):
    doc = nlp(text)
    aspects = [chunk.text for chunk in doc.noun_chunks]
    return aspects

# Apply the aspect extraction function to the train and test datasets
train_df["aspects"] = train_df["text"].apply(extract_aspects)
test_df["aspects"] = test_df["text"].apply(extract_aspects)

# Define a function to analyze the sentiment of each aspect
def analyze_aspect_sentiment(aspects):
    sentiment = []
    for aspect in aspects:
        doc = nlp(aspect)
        if doc._.polarity > 0:
            sentiment.append("positive")
        elif doc._.polarity < 0:
            sentiment.append("negative")
        else:
            sentiment.append("neutral")
    return sentiment

# Apply the aspect sentiment analysis function to the train and test datasets
train_df["aspect_sentiment"] = train_df["aspects"].apply(analyze_aspect_sentiment)
test_df["aspect_sentiment"] = test_df["aspects"].apply(analyze_aspect_sentiment)
