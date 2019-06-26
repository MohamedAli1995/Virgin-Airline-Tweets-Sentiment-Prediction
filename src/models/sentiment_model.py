from src.base.base_model import BaseModel
import tensorflow as tf


class SentimentModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.n_words = config.max_n_words
        self.x = None
        self.y = None
        self.train_step = None
        self.saver = None

        self.seq_len = None
        self.is_training = None
        self.keep_prob_lstm = None
        self.keep_prob_fc = None

        self.build_model()
        self.init_saver()

    def __word_embedding(self, x, vocab_size, embedding_size):
        # Create the embeddings
        with tf.name_scope("Embeddings"):
            embedding = tf.Variable(tf.random_uniform((vocab_size,
                                                       embedding_size), -1, 1))
            embedded_words = tf.nn.embedding_lookup(embedding, x)
            return embedded_words

    def __lstm_dropout_cell(self, hidden_size, keep_prob_out, keep_prob_recurrent):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
        dropout_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob_out,
                                                     output_keep_prob=keep_prob_out,
                                                     state_keep_prob=keep_prob_recurrent)
        return dropout_cell

    def __rnn(self, hidden_size, x, seq_len, keep_prob_out, keep_prob_recurrent):
        with tf.variable_scope(None, default_name='LSTM_layer'):
            lstm_cell = self.__lstm_dropout_cell(hidden_size, keep_prob_out, keep_prob_recurrent)
            outputs, last_state = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seq_len)
            return outputs

    def __init_weights(self, shape):
        initializer = tf.contrib.layers.xavier_initializer()
        init_xavier = initializer(shape)
        return tf.Variable(init_xavier)

    def __init_bias(self, shape):
        init_bias_vals = tf.constant(0.1, shape=shape)
        return tf.Variable(init_bias_vals)

    def __normal_full_layer(self, input_layer, size):
        input_size = int(input_layer.get_shape()[1])  # becase 0 is the number of training examples.
        W = self.__init_weights([input_size, size])
        b = self.__init_bias([size])
        return tf.matmul(input_layer, W) + b

    def __batch_norm(self, input_layer):
        return tf.contrib.layers.batch_norm(input_layer, activation_fn=tf.nn.leaky_relu, is_training=self.is_training)

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.seq_len = tf.placeholder(tf.int32, name='sequence_length')
        self.keep_prob_lstm_out = tf.placeholder(tf.float32, name='keep_prob_lstm_out')
        self.keep_prob_lstm_recurrent = tf.placeholder(tf.float32, name='keep_prob_lstm_recurrent')
        self.keep_prob_fc = tf.placeholder(tf.float32, name='keep_prob_fc')

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.int32, [None, None], name='inputs')

        with tf.name_scope('target'):
            self.y = tf.placeholder(tf.int32, [None, None], name='target')

        embedded_words = self.__word_embedding(self.x, self.n_words, 100)

        lstm_out = self.__rnn(100, embedded_words, self.seq_len, self.keep_prob_lstm_out, self.keep_prob_lstm_recurrent)
        # dropout_2 = tf.nn.dropout(lstm_1_out, self.keep_prob_fc)



        self.y_pred = self.__normal_full_layer(lstm_out[:, -1, :], 3)

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_pred))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                             global_step=self.global_step_tensor)
            correct_predictions = tf.equal(tf.argmax(tf.nn.softmax(self.y_pred), 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            self.predictions = tf.argmax(tf.nn.softmax(self.y_pred), 1)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)
