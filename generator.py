#!/usr/bin/env python3

import numpy as np

from config import (
    TEST_PREFIX, SAMPLE_SENTENCES_NUMBER, SENTENCE_ENDINGS, SENTENCE_MAX_LEN, SENTENCE_MIN_LEN
)

from global_variables import vocab, net

from utils import text_to_matrix


class CharGenerator:
    def __init__(self, prefix=TEST_PREFIX):
        self.out = None
        self.update_prefix(prefix)

    def __iter__(self):
        return self

    def __next__(self):
            char_vector = np.random.choice(range(len(vocab)), p=self.out)
            char = vocab[char_vector]
            self.out = net.run_step(text_to_matrix(char, vocab))
            return char

    def update_prefix(self, prefix):
        for i, char in enumerate(prefix):
            self.out = net.run_step(text_to_matrix(char, vocab), i == 0)


class SentenceGenerator:
    def __init__(self):
        self.char_generator = CharGenerator()

    def __iter__(self):
        return self

    def update_prefix(self, prefix):
        self.char_generator.update_prefix(prefix)

    def __next__(self):
        text = ''

        while len(text.strip()) < SENTENCE_MAX_LEN:
            text += next(self.char_generator)

            if text.endswith(SENTENCE_ENDINGS):
                if len(text.strip()) > SENTENCE_MIN_LEN:
                    break
                else:
                    text = ''

        return text.strip()


if __name__ == '__main__':
    generator = SentenceGenerator()
    print('\n\n'.join(next(generator) for i in range(SAMPLE_SENTENCES_NUMBER)))
