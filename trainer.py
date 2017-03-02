#!/usr/bin/env python3

import random
import time

import numpy as np

from config import CHECKPOINT_PATHS, BATCH_SIZE, TIME_STEPS, SAVE_RATE


from global_variables import vocab, sess, vec_vocab, saver, net, global_step


class Trainer:
    batch = np.zeros((BATCH_SIZE, TIME_STEPS, len(vocab)))
    batch_y = np.zeros((BATCH_SIZE, TIME_STEPS, len(vocab)))
    possible_batch_ids = range(vec_vocab.shape[0] - TIME_STEPS - 1)

    def __init__(self):
        self.loss = 0
        self.last_timestamp = 0
        self.trained_in_step = 0

    def train(self):
        global global_step

        self.last_timestamp = time.time()
        try:
            while True:
                global_step += 1
                batch_id = random.sample(self.possible_batch_ids, BATCH_SIZE)

                for j in range(TIME_STEPS):
                    ind1 = [k + j for k in batch_id]
                    ind2 = [k + j + 1 for k in batch_id]

                    self.batch[:, j, :] = vec_vocab[ind1, :]
                    self.batch_y[:, j, :] = vec_vocab[ind2, :]

                self.loss = net.train_batch(self.batch, self.batch_y)
                self.trained_in_step += 1

                if not global_step % SAVE_RATE:
                    self.log_and_save()
        except KeyboardInterrupt:
            self.log_and_save()
            print('bye')

    def log_and_save(self):
        saver.save(sess, CHECKPOINT_PATHS, global_step=global_step)
        now = time.time()
        print('batch: {step}\t loss: {loss:.5f}\t speed: {bps:.3f} batches/s'.format(
            step=global_step, loss=self.loss, bps=self.trained_in_step / (now - self.last_timestamp)
        ))
        self.trained_in_step = 0
        self.last_timestamp = now


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
