import re
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
import tensorflow as tf
DS_PATH = "/media/syrix/programms/projects/Virgin-Airline-Tweets-Sentiment-Prediction/data/Tweets.csv"
df = pd.read_csv(DS_PATH)

# Select only text & airline_sentiment fields.
df = df[["text", "airline_sentiment"]]

# Convert to lower.
df['text'] = df['text'].apply(lambda x: x.lower())
# Remove any symbols except @.
df['text'] = df['text'].apply((lambda x: re.sub('[^@a-zA-z\s]', '', x)))  #TODO: remove 0-9 same as @
# Remove anyword having @ in it, as it is a tag operator.
df['text'] = df['text'].apply(lambda x: re.sub('[\w]*[@+][\w]*[\s]*', '', x))

# Print how many positive, neutral and negative
print("Positive_count:", df[df['airline_sentiment'] == 'positive'].size)
print("Negative_count:", df[df['airline_sentiment'] == 'negative'].size)
print("Neutral_count:", df[df['airline_sentiment'] == 'neutral'].size)



# print(df[0:5][["text", "airline_sentiment"]])


max_fatures = 2500
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(df['text'].values)
X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X, maxlen=30)
print(X)
print(X.shape)
#
# # Longest sentence.
# #i  jetblue but i was on flt 277 from fll to sfo tke off was over 1 hr late flight div to phx amp got in 2 hrs late flight what will be done
#
# # print("a")
# # print(df['text'][8062])
# # X = pad_sequences(X)
# # print(X.shape)
# # Xencoded = np.array([np.array(xi) for xi in X])
# # tokenizer.fit_on_texts(["my name is ali ali"])
# # X = tokenizer.texts_to_sequences(["my"])
#
# # print(X)
#
#
# # model = Sequential()
# # model.add(Embedding(11, 5, input_length=2))
# # the model will take as input an integer matrix of size (batch, input_length).
# # the largest integer (i.e. word index) in the input should be
# # no larger than 999 (vocabulary size).
# # now model.output_shape == (None, 10, 64), where None is the batch dimension.
#
# # input_array = np.random.randint(11, size=(1, 2))
# # input_array = np.array([[1, 2]])
#
# # model.compile('rmsprop', 'mse')
# # output_array = model.predict(input_array)
# # assert output_array.shape == (1, 2, 5)
# # print(output_array)
#
# ######################
# ######################
# ######################
# max_fatures = 2000
# embed_size = 128
# lstm_size = 256
# tokenizer = Tokenizer(num_words=max_fatures, split=' ')
# tokenizer.fit_on_texts(df['text'].values)
# X = tokenizer.texts_to_sequences(df['text'].values)
#
# X = pad_sequences(X)
# Y = pd.get_dummies(df['airline_sentiment']).values
#
#
# ###############################
# ###############################
#
# max_fatures = 2000
# embed_size = 128
# hidden_size = 5
# is_training = True
# # keep_prob = 0.5
# sequence_len = X.shape[1]
#
# def __word_embedding(x, vocab_size, embedding_size):
#       # Create the embeddings
#     with tf.name_scope("Embeddings"):
#         embedding = tf.Variable(tf.random_uniform((vocab_size,
#                                                    embedding_size), -1, 1))
#         embedded_words = tf.nn.embedding_lookup(embedding, x)
#         return embedded_words
#
#
# def __lstm_dropout_cell(hidden_size, keep_prob):
#     lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
#     dropout_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob,
#                                                  output_keep_prob=keep_prob)
#     return dropout_cell
#
#
# def __rnn(hidden_size, x, seq_len, keep_prob):
#     with tf.variable_scope(None, default_name='LSTM_layer'):
#         lstm_cell = __lstm_dropout_cell(hidden_size, keep_prob)
#         outputs, last_state = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seq_len)
#         return outputs[:, -1, :]
#
#
# def __init_weights(shape):
#     initializer = tf.contrib.layers.xavier_initializer()
#     init_xavier = initializer(shape)
#     return tf.Variable(init_xavier)
#
#
# def __init_bias(shape):
#     init_bias_vals = tf.constant(0.1, shape=shape)
#     return tf.Variable(init_bias_vals)
#
#
# def __normal_full_layer(input_layer, size):
#     input_size = int(input_layer.get_shape()[1])  # becase 0 is the number of training examples.
#     W = __init_weights([input_size, size])
#     b = __init_bias([size])
#     return tf.matmul(input_layer, W) + b
#
#
# def __batch_norm(input_layer):
#     return tf.contrib.layers.batch_norm(input_layer, activation_fn=tf.nn.leaky_relu, is_training=is_training)
#
#
# seq_len = tf.placeholder(tf.int32, name='sequence_length')
#
#
#
# with tf.name_scope('input'):
#     x = tf.placeholder(tf.int32, [None, None], name='inputs')
#
# with tf.name_scope('target'):
#     y = tf.placeholder(tf.int32, [None, None], name='target')
#
# keep_prob_lstm = tf.placeholder(tf.float32, name='keep_prob_lstm')
# keep_prob_fc = tf.placeholder(tf.float32, name='keep_prob_fc')
#
# embedded_words = __word_embedding(x, max_fatures, embed_size)
#
# lstm_1_out = __rnn(hidden_size, embedded_words, seq_len, keep_prob_lstm)
#
# full_layer_1 = __normal_full_layer(lstm_1_out, 128)
# batch_norm_1 = __batch_norm(full_layer_1)
# dropout_1 =tf.nn.dropout(batch_norm_1, keep_prob_fc)
#
# output = __normal_full_layer(dropout_1, 3)
#
#
#
# sess = tf.Session()
# sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
#
# feed_dict = {x: X[0:1], seq_len: X.shape[1], keep_prob_lstm: 0.2, keep_prob_fc:0.5}
# ret = sess.run(output, feed_dict=feed_dict)
# print(ret)
#
#
#
# # # Building Model
# # with tf.name_scope('Inputs'):
# #     x = tf.placeholder(tf.int32, [None, None], name='inputs')
# # with tf.name_scope('Labels'):
# #     y = tf.placeholder(tf.int32, [None, None], name='labels')
# #
# # keep_prob = tf.placeholder(tf.float32, name='keep_prob')
# #
# # # Create the embeddings
# # with tf.name_scope("Embeddings"):
# #     embedding = tf.Variable(tf.random_uniform((max_fatures,
# #                                                embed_size), -1, 1))
# #     embed = tf.nn.embedding_lookup(embedding, x)
# #
# # # Build the RNN layers
# # with tf.name_scope("LSTM_Layers"):
# #     lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
# #     drop = tf.contrib.rnn.DropoutWrapper(lstm,
# #                                      output_keep_prob=keep_prob)
#