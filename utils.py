import numpy as np


def text_to_matrix(data, vocab):
    vec = np.zeros((len(data), len(vocab)))

    for i, char in enumerate(data):
        v = [0.0] * len(vocab)
        v[vocab.index(char)] = 1.0
        vec[i, :] = v

    return vec


def decode_embed(array, vocab):
    return vocab[array.index(1)]


def get_last_step(checkpoint_names):
    return int(checkpoint_names[-1].split('-')[-1]) + 1 if checkpoint_names else 0
