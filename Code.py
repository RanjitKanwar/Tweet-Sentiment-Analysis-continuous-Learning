import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd

# Load the Spacy English model
nlp = spacy.load("en_core_web_sm")

# Load the train and test datasets
train_df = pd.read_csv("train_tweet.csv")
test_df = pd.read_csv("test_tweet.csv")

# Define the vectorizer and classifier
vectorizer = CountVectorizer()
classifier = MultinomialNB()

# Preprocess the data
train_tweets = [nlp(tweet) for tweet in train_df["tweet"]]
test_tweets = [nlp(tweet) for tweet in test_df["tweet"]]

# Extract the aspects
train_aspects = train_df["aspects"]
test_aspects = test_df["aspects"]

# Extract the sentiment labels
train_sentiment = train_df["sentiment"]
test_sentiment = test_df["sentiment"]

# Extract the TextBlob scores for each tweet
train_scores = [TextBlob(tweet.text).sentiment.polarity for tweet in train_tweets]
test_scores = [TextBlob(tweet.text).sentiment.polarity for tweet in test_tweets]

# Combine the TextBlob scores with the aspect and sentiment labels
train_data = [[train_scores[i], train_aspects[i], train_sentiment[i]] for i in range(len(train_tweets))]
test_data = [[test_scores[i], test_aspects[i], test_sentiment[i]] for i in range(len(test_tweets))]

# Vectorize the data
X_train = vectorizer.fit_transform([" ".join([str(item) for item in tweet]) for tweet in train_data])
X_test = vectorizer.transform([" ".join([str(item) for item in tweet]) for tweet in test_data])

# Fit the classifier
classifier.fit(X_train, train_sentiment)

# Make predictions on the test set
test_predictions = classifier.predict(X_test)

# Print the classification report
print(classification_report(test_sentiment, test_predictions))

# Count the number of times each aspect appears in the dataset
aspect_counts = df["aspects"].value_counts()

# Plot the results
plt.bar(aspect_counts.index, aspect_counts.values)

# Set the plot title and axis labels
plt.title("Aspect frequency in training dataset")
plt.xlabel("Aspect")
plt.ylabel("Frequency")

# Show the plot
plt.show()
