import nltk
from nltk.corpus import stopwords
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle as pk
import string
from typing import Dict, Tuple, List, Set

# Shared constants
TRAIN_TWEET_PATH = "train_tweet.csv"
labels = ["negative", "neutral", "positive"]

# Keywords for aspects
aspect_keywords = {
    "flight delays": ["delay", "cancel", "divert", "late", "wait", "postpone"],
    "on-time performance": ["on time", "early", "late", "delayed", "timely"],
    "baggage handling": ["baggage", "luggage", "lost", "damage", "stolen", "missing"],
    "customer service": ["service", "help", "assistance", "complaint", "support", "response"],
    "in-flight experience": ["experience", "comfort", "food", "entertainment", "seat", "cabin"],
    "pricing": ["price", "cost", "fare", "expensive", "cheap", "affordable"],
    "safety": ["safety", "security", "crash", "accident", "emergency", "evacuation"],
    "loyalty programs": ["loyalty", "points", "miles", "rewards", "perks", "benefits"]
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

def str_to_list(instring: str) -> Tuple[list[str], list[str]]:
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
    mentioned_aspects = []
    for aspect, keywords in aspect_keywords.items():
        for keyword in keywords:
            if keyword in tweet.lower():
                mentioned_aspects.append(aspect)
                break

    # Return the list of words and mentioned aspects
    return wordlist, mentioned_aspects
