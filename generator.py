#!/usr/bin/env python3

import numpy as np

from config import TEST_PREFIX, LEN_TEST_TEXT, SENTENCE_ENDINGS

from global_variables import vocab, net

from utils import text_to_matrix


class SentenceGenerator:
    def __init__(self, prefix=TEST_PREFIX):
        self.update_prefix(prefix)

    def __iter__(self):
        return self

    def update_prefix(self, prefix=TEST_PREFIX):
        for i, char in enumerate(prefix):
            self.out = net.run_step(text_to_matrix(char, vocab), i == 0)

    def __next__(self):
        text = ''

        while len(text) < 5000:
            element = np.random.choice(range(len(vocab)), p=self.out)
            text += vocab[element]

            if text.endswith(SENTENCE_ENDINGS):
                if len(text) > 5:
                    break
                else:
                    text = ''
            self.out = net.run_step(text_to_matrix(vocab[element], vocab))

        return text


if __name__ == '__main__':
    s = SentenceGenerator()
    print('\n\n'.join(next(s) for i in range(LEN_TEST_TEXT)))
