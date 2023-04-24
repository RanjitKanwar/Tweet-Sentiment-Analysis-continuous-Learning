import csv
import pickle as pk
import numpy as np
import util

# column indices
TWEET_COL = 0
SENTIMENT_COL = 1
ASPECT_COL = 2

# Load vocabulary
with open(util.vocab_list_path, "rb") as f:
    vocab_list = pk.load(f)
vocab_lookup = {w:i for i, w in enumerate(vocab_list)}

# Get number of labels
labels = ["negative", "neutral", "positive"]
num_labels = len(labels)

# Initialize sentiment counts
sentiment_counts = [0] * num_labels

# Initialize count matrix
counts = np.zeros((num_labels, len(vocab_list)))
# Process training data
with open(util.TRAIN_TWEET_PATH, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        sentiment = labels.index(row[SENTIMENT_COL])
        tweet_words = row[TWEET_COL].strip().split()
        aspects = row[ASPECT_COL].strip().split(';')

        # Update counts for each aspect
        for aspect in aspects:
            aspect_counts = np.zeros(len(vocab_list))
            for word in tweet_words:
                if word in vocab_list:
                    aspect_counts[vocab_list.index(word)] += 1
            counts[sentiment, vocab_list.index(aspect)] += np.sum(aspect_counts)

        # Update counts for all words
        for word in tweet_words:
            if word in vocab_list:
                counts[sentiment, vocab_list.index(word)] += 1

        # Update sentiment counts
        sentiment_counts[sentiment] += 1

# Calculate log-frequencies for each aspect and sentiment
aspect_freqs = np.zeros((len(labels), len(vocab_list)))
for aspect in range(len(vocab_list)):
    total = np.sum(counts[:, aspect])
    for sentiment in range(len(labels)):
        aspect_freqs[sentiment][aspect] = np.log((counts[sentiment][aspect] + 1) / (total + len(vocab_list)))

sentiment_freqs = np.zeros((len(labels)))
for sentiment in range(len(labels)):
    total = np.sum(counts[sentiment])
    sentiment_freqs[sentiment] = np.log(total / np.sum(counts[:, sentiment]))

# Save out frequencies
with open(util.freq_path, "wb") as f:
    pk.dump((aspect_freqs, sentiment_freqs), f)

# Find the most positive and negative aspects
diff = aspect_freqs[util.labels.index("positive")] - aspect_freqs[util.labels.index("negative")]
indices = diff.argsort()

print("Most positive aspects:")
for i in range(1, 11):
    index = indices[-i]
    print(f"{vocab_list[index]} ({np.exp(aspect_freqs[util.labels.index('positive')][index])})")

print("Most negative aspects:")
for i in range(1, 11):
    index = indices[i - 1]
    print(f"{vocab_list[index]} ({np.exp(aspect_freqs[util.labels.index('negative')][index])})")
