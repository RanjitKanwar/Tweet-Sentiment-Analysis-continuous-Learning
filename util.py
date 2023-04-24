import nltk
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from collections import Counter
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle as pk
import string
from typing import Dict, Tuple

# Shared constants
TRAIN_TWEET_PATH = "train_tweet.csv"
freq_path = "frequencies_tweet.pkl"
vocab_list_path = "vocablist_tweet.pkl"
frequencies_path = "frequencies_tweet.pkl"
labels = ["negative", "neutral", "positive"]

# Keywords for aspects
ASPECT_KEYWORDS = {
    "flight delays": ["delay", "cancel", "divert"],
    "on-time performance": ["on time", "delay", "early", "late"],
    "baggage handling": ["baggage", "luggage", "lost", "damage"],
    "customer service": ["service", "help", "assistance", "complaint"],
    "in-flight experience": ["experience", "comfort", "food", "entertainment"],
    "pricing": ["price", "cost", "fare", "expensive"],
    "safety": ["safety", "security", "crash", "accident"],
    "loyalty programs": ["loyalty", "points", "miles", "rewards"]
}
# Initialization
ignore = {
    "united",
    "usairways",
    "americanair",
    "southwestair",
    "jetblue",
    "virginamerica",
}
# Put the stop words in a set
stop_words = set(stopwords.words("english"))

# Add the airline names
stop_words.update(ignore)

# Create a translator that deletes the punctuation and digits
translator = str.maketrans("", "", string.punctuation + string.digits)

# Define aspects to look for
aspects = {
    "flight delays",
    "on-time performance",
    "baggage handling",
    "customer service",
    "in-flight experience",
    "pricing",
    "safety",
    "loyalty programs",
}


def str_to_list(instring: str) -> tuple[list[str], list[str]]:
    global stop_words, aspects, translator

    # Get rid of leading and trailing whitespace
    tweet = instring.strip()

    # Get rid of punctuation and digits
    tweet = tweet.translate(translator)

    # Tokenize the tweet into words
    tweetwords = word_tokenize(tweet)

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    wordlist = [lemmatizer.lemmatize(w.lower()) for w in tweetwords]

    # Put non-stop words in a list
    wordlist = [w for w in wordlist if w not in stop_words]

    # Identify aspects mentioned in the tweet
    mentioned_aspects = [aspect for aspect in aspects if aspect in tweet.lower()]

    # Return the list of words and mentioned aspects
    return wordlist, mentioned_aspects


def load_vocab() -> Dict[str, int]:
    global vocab_list_path

    with open(vocab_list_path, "rb") as f:
        vocab_list = pk.load(f)

    # Create a dictionary that maps words to their indices in the vocabulary
    vocab_lookup = {word: index for index, word in enumerate(vocab_list)}

    return vocab_lookup


def counts_for_wordlist(wordlist: list[str], vocab_lookup: dict[str, int], aspects: set[str]) -> tuple[
    np.array, list[str]]:
    # Count the occurrences of each word in the wordlist
    word_counts = Counter(wordlist)

    # Create an empty array as long as the vocabulary
    count_vec = np.zeros(len(vocab_lookup))

    # Use the Counter to populate the count_vec array
    count_vec[:] = [word_counts.get(word, 0) for word in vocab_lookup]

    # Identify aspects mentioned in the tweet
    mentioned_aspects = [aspect for aspect in aspects if aspect in ' '.join(wordlist)]

    # Did this tweet have no words in the vocabulary?
    if count_vec.sum() == 0:
        # Let the caller know that the vector would have been zero
        return None, mentioned_aspects
    else:
        # Return the counts and mentioned aspects
        return count_vec, mentioned_aspects
