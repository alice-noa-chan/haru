from __future__ import annotations

import itertools

import sentencepiece as spm

import config
from data_utils import iter_text_records, prepare_text_for_tokenizer


def sentence_iterator():
    """Stream source text to SentencePiece without loading the corpus into RAM."""

    records = iter_text_records()
    texts = (prepare_text_for_tokenizer(record.text) for record in records)

    if config.TOKENIZER_MAX_SENTENCES > 0:
        yield from itertools.islice(texts, config.TOKENIZER_MAX_SENTENCES)
    else:
        yield from texts


def main() -> None:
    config.TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    model_prefix = config.TOKENIZER_MODEL_PATH.with_suffix("")

    print("Training the SentencePiece tokenizer...", flush=True)
    print(f"vocab_size={config.TOKENIZER_VOCAB_SIZE:,}", flush=True)
    print(f"data_dir={config.DATA_DIR}", flush=True)

    spm.SentencePieceTrainer.train(
        sentence_iterator=sentence_iterator(),
        model_prefix=str(model_prefix),
        vocab_size=config.TOKENIZER_VOCAB_SIZE,
        model_type=config.TOKENIZER_MODEL_TYPE,
        character_coverage=config.TOKENIZER_CHARACTER_COVERAGE,
        byte_fallback=config.TOKENIZER_BYTE_FALLBACK,
        max_sentence_length=config.TOKENIZER_MAX_SENTENCE_LENGTH,
        input_sentence_size=config.TOKENIZER_MAX_SENTENCES,
        shuffle_input_sentence=True,
        num_threads=config.TOKENIZER_NUM_THREADS,
        hard_vocab_limit=False,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        pad_id=config.PAD_ID,
        unk_id=config.UNK_ID,
        bos_id=config.BOS_ID,
        eos_id=config.EOS_ID,
        user_defined_symbols=[config.NEWLINE_TOKEN, config.NEWLINE_ESCAPE_TOKEN],
    )

    print(f"Saved: {config.TOKENIZER_MODEL_PATH}", flush=True)
    print(f"Saved: {config.TOKENIZER_VOCAB_PATH}", flush=True)


if __name__ == "__main__":
    main()
