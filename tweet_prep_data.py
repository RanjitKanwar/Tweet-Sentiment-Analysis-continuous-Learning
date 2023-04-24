import csv
import util
import numpy as np
import re

# column indices
SENTIMENT_COL = 1
CONFIDENCE_COL = 2
TWEET_COL = 10

# Probabilities
TEST_P = 0.1

# Confidence required
REQUIRED_CONFIDENCE = 0.5

# The file we are reading the tweets from
INPATH = "Tweets.csv"

# Create an output file for each phase
phases = ["train", "test"]
writers = {}
for phase in phases:
    f = open(f"{phase}_tweet.csv", "w", newline="\n")
    writer = csv.writer(f)
    writers[phase] = writer

# Read all the tweets
with open(INPATH, "r") as f:
    reader = csv.reader(f)

    # Skip header
    next(reader)

    # Count the words as we go
    count_dict = {}

    # Count how many are discarded for low confidence
    discarded_count = 0
    saved_count = 0
    for row in reader:
        # Get the sentiment, confidence, and tweet text from the row
        sentiment = row[SENTIMENT_COL]
        confidence = float(row[CONFIDENCE_COL])
        tweet = row[TWEET_COL]

        # Skip the tweet if the confidence is below the threshold
        if confidence < REQUIRED_CONFIDENCE:
            discarded_count += 1
            continue

        # Tokenize the tweet and identify mentioned aspects
        wordlist, mentioned_aspects = util.str_to_list(tweet)

        # If no aspects are mentioned, skip the tweet
        if not mentioned_aspects:
            discarded_count += 1
            continue

        # Which file should it go into?
        r = np.random.rand()
        if r < TEST_P:
            destination = "test"
        else:
            destination = "train"

        # Get sentiment
        sentiment = util.labels.index(row[SENTIMENT_COL])

        # Write it out
        writers[destination].writerow([tweet, row[SENTIMENT_COL], mentioned_aspects])
        saved_count += 1

        # Count the words in the list
        for w in wordlist:
            if w not in count_dict:
                count_dict[w] = 1
            else:
                count_dict[w] += 1

print(f"Kept {saved_count} rows, discarded {discarded_count} rows")
