VOCAB_PATH = 'data/vocab.npy'
VEC_VOCAB_PATH = 'data/vec_vocab.npy'

CHECKPOINT_DIR = 'data/checkpoints'
CHECKPOINT_PATHS = 'data/checkpoints/checkpoint.ckpt'

TEXT_PATH = 'data/rationality.txt'


LSTM_SIZE = 256  # 128
NUM_LAYERS = 2
BATCH_SIZE = 64  # 128
TIME_STEPS = 100  # 50
STATE_IS_TUPLE = False
LEARNING_RATE = 0.003
SAVE_RATE = 10

SAMPLE_SENTENCES_NUMBER = 5
SENTENCE_MIN_LEN = 5
SENTENCE_MAX_LEN = 5000
SENTENCE_ENDINGS = [
    '\n',
    '. ',
    '!',
    '?',
]

TEST_PREFIX = '\n'


BOT_TOKEN = '123'
BOT_LOG_CHANNEL_ID = -123

NOT_A_TEXT_REPLIES = [
    'Is it a question?',
    "I don't understand",
    '👀',
    '🙄',
    '🙈'
]
