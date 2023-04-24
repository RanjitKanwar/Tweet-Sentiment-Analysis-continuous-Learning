import csv
import pickle as pk
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import util


# Read in the vocabulary
with open(util.vocab_list_path, "rb") as f:
    vocab_list = pk.load(f)

# Convert dictionary (str -> index) for faster lookup
vocab_lookup = {}
for i, word in enumerate(vocab_list):
    vocab_lookup[word] = i

# Read in the log of the word and sentiment frequencies
with open(util.frequencies_path, "rb") as f:
    log_word_frequencies, log_sentiment_frequencies = pk.load(f)

# Transpose the word frequencies to match the shape of the vectors
log_word_frequencies = log_word_frequencies.T

# Check that the shapes of the arrays match the expected values
assert log_word_frequencies.shape == (3, util.VECLEN), "log_word_frequencies is the wrong shape"
assert log_sentiment_frequencies.shape == (3,), "log_sentiment_frequencies is the wrong shape"

baseline_accuracy = freq_of_most_common * 100.0
print(f'The most common sentiment is "{most_common_sentiment}".')
print(f'The baseline accuracy is {baseline_accuracy:.1f}%.')
print('This means that our sentiment analysis model needs to achieve an accuracy')
print('higher than the baseline to be considered useful.')

# Gather ground truth and predictions
gt_sentiments = []
predicted_sentiments = []
prediction_confidences = []

# Step through the test.csv file
with open("test_tweet.csv", "r") as f:
    reader = csv.reader(f)

    skipped_tweet_count = 0
    for row in reader:

        # Check to see if the row has two entries
        if len(row) != 2:
            continue

        # Get the tweet and its ground truth sentiment
        tweet = row[0]
        sentiment = int(row[1])
        gt_sentiments.append(sentiment)

        # Tokenize the tweet and identify mentioned aspects
        wordlist, mentioned_aspects = util.str_to_list(tweet)

        # Convert the tweet to a vector using the vocabulary and lookup dictionary
        counts = np.zeros(util.VECLEN)
        for word in wordlist:
            if word in vocab_lookup:
                index = vocab_lookup[word]
                counts[index] += 1

        # Normalize the vector
        total_count = counts.sum()
        if total_count > 0:
            counts /= total_count

        # Calculate the log-probability of the tweet belonging to each sentiment class
        log_probs = log_word_frequencies @ counts + log_sentiment_frequencies

        # Use the class with the highest log-probability as the predicted sentiment
        predicted_sentiment = np.argmax(log_probs)
        predicted_sentiments.append(predicted_sentiment)

        # Calculate the confidence in the prediction
        prediction_confidence = np.exp(log_probs[predicted_sentiment])
        prediction_confidences.append(prediction_confidence)


print(f"Skipped {skipped_tweet_count} rows for having none of the common words")

# Convert gathered data into numpy arrays
gt = np.array(gt_sentiments)
predictions = np.array(predicted_sentiments)
confidence = np.array(prediction_confidences)

# Show some basic statistics
tweet_count = len(gt)  ## Your code
correct_count = np.sum(gt == predictions)  ## Your code
print(
    f"{tweet_count} lines analyzed, {correct_count} correct ({100.0 * correct_count/tweet_count:.1f}% accuracy)"
)
cm = confusion_matrix(gt, predictions)  ## Your code
print(f"Confusion: \n{cm}")

# Save out a confusion matrix plot
fig, ax = plt.subplots()
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=util.labels)
cm_display.plot(ax=ax, cmap="Blues", colorbar=False)
fig.savefig("confusion_tweet.png")
print('Wrote confusion matrix as "confusion_tweet.png"')

# Plot how many you get right vs confidence thresholds
steps = 32
thresholds = np.linspace(0.33, 1.0, steps)
correct_ratio = np.zeros(steps)
confident_ratio = np.zeros(steps)

for i in range(steps):
    threshold = thresholds[i]
    ## Your code here
    if np.sum(confidence > threshold) == 0:
        correct_ratio[i] = 1
    else:
        correct_ratio[i] = np.sum(
            (gt == predictions) & (confidence > threshold)
        ) / np.sum(confidence > threshold)
    confident_ratio[i] = np.sum(confidence > threshold) / len(confidence)

# Make a plot
fig, ax = plt.subplots()
ax.set_title("Confidence and Accuracy Are Correlated")
ax.plot(
    thresholds, correct_ratio, "blue", linewidth=0.8, label="Accuracy Above Threshod"
)
ax.set_xlabel("Confidence Threshold")
ax.yaxis.set_major_formatter(lambda x, pos: f"{x*100.0:.0f}%")
ax.hlines(
    freq_of_most_common,
    0.33,
    1,
    "blue",
    linestyle="dashed",
    linewidth=0.8,
    label=f"Accuracy Guessing {most_common_sentiment}",
)
ax.plot(
    thresholds,
    confident_ratio,
    "r",
    linestyle="dashed",
    linewidth=0.8,
    label="Tweets scoring above threshold",
)
ax.legend()
fig.savefig("confidence_tweet.png")
