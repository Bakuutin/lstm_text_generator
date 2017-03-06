#!/usr/bin/env python3

from pprint import pprint
import random
import time
import traceback

import telepot
import telepot.namedtuple
import language_check

from config import BOT_TOKEN, BOT_LOG_CHANNEL_ID, NOT_A_TEXT_REPLIES
from global_variables import vocab_set
from generator import SentenceGenerator

lang_tool = language_check.LanguageTool('en-US')


class InvalidText(Exception):
    pass


class NeuroBot(telepot.Bot):
    log_channel_id = BOT_LOG_CHANNEL_ID
    not_a_text_replies = NOT_A_TEXT_REPLIES

    def __init__(self, *args, **kwargs):
        self.generator = SentenceGenerator()
        super().__init__(*args, **kwargs)

    def log(self, text, disable_notification=True, **kwargs):
        self.sendMessage(
            self.log_channel_id, text, disable_notification=disable_notification, parse_mode='Markdown', **kwargs
        )

    def log_exception(self, e, **kwargs):
        self.log('```{}```'.format(traceback.format_exc()))

    def forward_to_log(self, msg, disable_notification=True, messages_number=2):
        for i in range(msg['message_id'] - messages_number, msg['message_id']):
            self.forwardMessage(
                self.log_channel_id, msg['chat']['id'], i + 1,
                disable_notification=disable_notification
            )

    def handle(self, msg):
        try:
            pprint(msg)
            content_type, chat_type, chat_id = telepot.glance(msg)
            if chat_type == 'channel' or 'forward_from' in msg or 'edit_date' in msg:
                return

            reply = self.get_reply(msg, content_type)

            self.forward_to_log(self.sendMessage(chat_id, reply))
        except Exception as e:
            self.log_exception(e)

    @staticmethod
    def prepare_for_lstm(text):
        text = text.strip()

        if text and text[~0] not in {'?', '.', '!'}:
            text += '?'

        if text.startswith('/'):
            text = text[1:]

        return ''.join(c for c in text.lower() if c in vocab_set)

    def get_reply(self, msg, content_type):
        try:
            if content_type != 'text':
                raise InvalidText

            prepared_text = self.prepare_for_lstm(msg['text'])

            if not prepared_text:
                raise InvalidText

            self.generator.update_prefix(prepared_text)
            return self.post_process(next(self.generator))
        except InvalidText:
            return random.choice(self.not_a_text_replies)

    @staticmethod
    def post_process(text):
        text = text.strip().replace(' i ', ' I ')
        return language_check.correct(text, lang_tool.check(text))


if __name__ == '__main__':
    bot = NeuroBot(BOT_TOKEN)
    bot.message_loop()
    print('Listening...')

    while True:
        time.sleep(10)
