import logging
import os

import numpy as np
import tensorflow as tf

from config import TEXT_FILE, CHECKPOINT_DIR, VOCAB_PATH, VEC_VOCAB_PATH
from utils import text_to_matrix, get_last_step
from lstm import LSTMNetwork


logging.getLogger('tensorflow').setLevel(logging.ERROR)


if not os.path.exists(VOCAB_PATH) or not os.path.exists(VEC_VOCAB_PATH):
    with open(TEXT_FILE, 'r') as f:
        data = f.read().lower()
    vocab = sorted(set(data))
    vec_vocab = text_to_matrix(data, vocab)

    np.save(VOCAB_PATH, vocab)
    np.save(VEC_VOCAB_PATH, vec_vocab)

else:
    vocab = list(np.load(VOCAB_PATH))
    vec_vocab = np.load(VEC_VOCAB_PATH)


net = LSTMNetwork(vocab_size=len(vocab))

saver = tf.train.Saver()

init = tf.global_variables_initializer()

session_manager = tf.train.SessionManager()
sess = session_manager.prepare_session(master='', init_op=init, saver=saver, checkpoint_dir=CHECKPOINT_DIR)

global_step = get_last_step(saver.last_checkpoints)

net.set_session(sess)
