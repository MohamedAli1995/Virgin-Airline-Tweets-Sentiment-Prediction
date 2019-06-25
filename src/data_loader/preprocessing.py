import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
from src.utils.dirs import create_dirs
import pickle

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


def text_to_seq(values,  tokenizer_pickle_path, max_features=2000, create_tokenizer=True):
    if create_tokenizer:
        create_dirs([tokenizer_pickle_path])

        tokenizer = Tokenizer(num_words=max_features, split=' ')
        tokenizer.fit_on_texts(values)

        x = tokenizer.texts_to_sequences(values)
        x = pad_sequences(x, maxlen=30)

        # saving
        with open(tokenizer_pickle_path+'tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return x
    else:
        # loading
        with open(tokenizer_pickle_path + 'tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            x = tokenizer.texts_to_sequences(values)
            x = pad_sequences(x, maxlen=30)
            return x







def one_hot_encoding(labels, num_classes, dict):
    all_one_hot_encoded = []
    for label in labels:
        one_hot_encoded = [0] * num_classes
        one_hot_encoded[int(dict[label])] = 1
        all_one_hot_encoded.append(one_hot_encoded)
    return np.asanyarray(all_one_hot_encoded)
