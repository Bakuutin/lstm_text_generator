#!/usr/bin/env python3

from pprint import pprint
import random
import time
import telepot
import telepot.namedtuple

import traceback

import telepot

from config import BOT_TOKEN, BOT_LOG_CHANNEL_ID

from global_variables import vocab

from generator import SentenceGenerator


logging_bot = telepot.SpeakerBot(BOT_TOKEN)


def log(text, disable_notification=True, **kwargs):
    logging_bot.sendMessage(
        BOT_LOG_CHANNEL_ID, text, disable_notification=disable_notification, parse_mode='Markdown', **kwargs
    )


def log_exception(e, **kwargs):
    log('```{}```'.format(traceback.format_exc()))


def forward_to_log(msg, disable_notification=True, messages_number=2):
    for i in range(msg['message_id'] - messages_number, msg['message_id']):
        logging_bot.forwardMessage(
            BOT_LOG_CHANNEL_ID, msg['chat']['id'], i + 1, disable_notification=disable_notification
        )


not_a_text_replies = [
    'Is it a question?',
    "I don't understand",
    '👀',
    '🙄',
    '🙈'
]

vocabulary_set = set(vocab)


def handle(msg):
    try:
        pprint(msg)
        content_type, chat_type, chat_id = telepot.glance(msg)
        if chat_type == 'channel' or {'forward_from', 'edit_date'} & set(msg):
            return

        if content_type != 'text':
            reply = random.choice(not_a_text_replies)
        else:
            msg['text'] = msg['text'].lower()
            if msg['text'].startswith('/'):
                msg['text'] = msg['text'][1:]

            if all(c not in vocabulary_set for c in msg['text']):
                reply = random.choice(not_a_text_replies)
            else:
                if set(msg['text']) < set(vocab):
                    if not msg['text'].endswith('?'):
                        msg['text'] += '?'
                    generator.update_prefix(msg['text'])
                reply = next(generator)

        forward_to_log(bot.sendMessage(chat_id, reply))
    except Exception as e:
        log_exception(e)


bot = telepot.Bot(BOT_TOKEN)
generator = SentenceGenerator()

if __name__ == '__main__':
    bot.message_loop(handle)
    print('Listening...')

    while True:
        time.sleep(10)
