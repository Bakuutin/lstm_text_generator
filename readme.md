# Installation

    pip install -Ur requirements.txt
    mkdir -p data/checkpoints
    cp config.example.py config.py

*Optional:* Add your telegram key to `config.py`.

# Train on the new text

Set proper `TEXT_PATH`.

Delete `checkpoints/*`, `vocab.npy` and `vec_vocab.npy` from `data` directory.

Run `./trainer.py`.

Press `ctrl+c` to stop training.

# Run

Use `./generator.py` to check the texts.

For bot mode you must set `BOT_TOKEN` and `BOT_LOG_CHANNEL_ID`.
Run it via `./bot.py`.

# Pre trained net

Feel free to use [this snapshot of data](https://drive.google.com/file/d/0BwKD7s64TX1KeVZucnp5NzkxZnM/view).

The text is written by Eliezer Yudkowsky on human rationality and irrationality in cognitive science. You can find the original on the [LessWrong](http://lesswrong.com/).
