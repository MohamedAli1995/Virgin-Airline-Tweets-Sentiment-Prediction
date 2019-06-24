import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re


def preprocess_text(path):
    df = pd.read_csv(path)
    # Select only text & airline_sentiment fields.
    df = df[["text", "airline_sentiment"]]
    # Convert to lower.
    df['text'] = df['text'].apply(lambda x: x.lower())
    # Remove any symbols except @.
    df['text'] = df['text'].apply((lambda x: re.sub('[^@a-zA-z\s]', '', x)))  # TODO: remove 0-9 same as @
    # Remove anyword having @ in it, as it is a tag operator.
    df['text'] = df['text'].apply(lambda x: re.sub('[\w]*[@+][\w]*[\s]*', '', x))
    return df


def text_to_seq(values, max_features, tokenizer_pickle_path=None):
    if tokenizer_pickle_path == None:
        tokenizer = Tokenizer(num_words=max_features, split=' ')
        tokenizer.fit_on_texts(values)
        x = tokenizer.texts_to_sequences(values)
        x = pad_sequences(x)
        return x


def one_hot_encoding(labels, num_classes, dict):
    all_one_hot_encoded = []
    for label in labels:
        one_hot_encoded = [0] * num_classes
        one_hot_encoded[int(dict[label])] = 1
        all_one_hot_encoded.append(one_hot_encoded)
    return np.asanyarray(all_one_hot_encoded)
