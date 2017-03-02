import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import MultiRNNCell, BasicLSTMCell

from config import LSTM_SIZE, NUM_LAYERS, STATE_IS_TUPLE, LEARNING_RATE


class LSTMNetwork:
    learning_rate = tf.constant(LEARNING_RATE)

    def __init__(self, vocab_size, name='char_rnn_network'):
        self.vocab_size = vocab_size
        self.scope = name
        self.last_state = np.zeros((NUM_LAYERS * 2 * LSTM_SIZE,))

        with tf.variable_scope(self.scope):
            self.xinput = tf.placeholder(tf.float32, shape=(None, None, self.vocab_size), name='xinput')
            self.lstm_init_value = tf.placeholder(tf.float32, shape=(
                None, NUM_LAYERS * 2 * LSTM_SIZE), name='lstm_init_value'
            )

            self.lstm = MultiRNNCell(
                [BasicLSTMCell(LSTM_SIZE, forget_bias=1.0, state_is_tuple=STATE_IS_TUPLE)] * NUM_LAYERS,
                state_is_tuple=STATE_IS_TUPLE
            )

            outputs, self.lstm_new_state = tf.nn.dynamic_rnn(
                self.lstm, self.xinput, initial_state=self.lstm_init_value, dtype=tf.float32
            )

            # Linear activation (FC layer on top of the LSTM net)
            self.rnn_out_W = tf.Variable(tf.random_normal((LSTM_SIZE, self.vocab_size), stddev=0.01))
            self.rnn_out_B = tf.Variable(tf.random_normal((self.vocab_size, ), stddev=0.01))

            outputs_reshaped = tf.reshape(outputs, [-1, LSTM_SIZE])
            network_output = (tf.matmul(outputs_reshaped, self.rnn_out_W) + self.rnn_out_B)

            batch_time_shape = tf.shape(outputs)
            self.final_outputs = tf.reshape(
                tf.nn.softmax(network_output),
                (batch_time_shape[0], batch_time_shape[1], self.vocab_size)
            )

            self.y_batch = tf.placeholder(tf.float32, (None, None, self.vocab_size))
            y_batch_long = tf.reshape(self.y_batch, [-1, self.vocab_size])

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=network_output, labels=y_batch_long
            ))
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate, 0.9).minimize(self.cost)

    def set_session(self, session):
        self.session = session

    def run_step(self, x, init_zero_state=False):
        if init_zero_state:
            init_value = np.zeros((NUM_LAYERS * 2 * LSTM_SIZE,))
        else:
            init_value = self.last_state

        out, next_state = self.session.run(
            [self.final_outputs, self.lstm_new_state],
            feed_dict={self.xinput: [x], self.lstm_init_value: [init_value]}
        )

        self.last_state = next_state[0]

        return out[0][0]

    def train_batch(self, xbatch, ybatch):
        init_value = np.zeros((xbatch.shape[0], NUM_LAYERS * 2 * LSTM_SIZE))

        cost, _ = self.session.run([self.cost, self.train_op], feed_dict={
            self.xinput: xbatch, self.y_batch: ybatch, self.lstm_init_value: init_value
        })

        return cost
