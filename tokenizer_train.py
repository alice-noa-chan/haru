"""Train the SentencePiece tokenizer on a proportional sample of the corpus.

Sampling is written to one file and SentencePiece reads that file directly.
Handing it a Python generator instead pushes every record across the
Python/C++ boundary one string at a time: the same vocabulary that trains in
about a minute from a file did not finish in four hours from an iterator over
4,042,834 records.

The sample is drawn proportionally from every corpus. Streaming records in file
order and stopping at a cap does not work either, because the files are read in
sorted order and the first one alone holds 2,003,542 records. That produced a
v2.0 tokenizer trained on 100% stories and 0% of the encyclopaedic and web text
that is most of the corpus, and it looked entirely successful: training
finished, the vocabulary filled, the files saved.
"""

from __future__ import annotations

import time

import sentencepiece as spm

import config
from data_utils import iter_data_files, prepare_text_for_tokenizer

SAMPLE_FILE = "tokenizer_sample.txt"


def line_counts(paths: list) -> dict:
    """Count records per corpus so the sample can be drawn in proportion."""

    counts = {}
    for path in paths:
        with path.open("rb") as handle:
            counts[path] = sum(1 for _ in handle)
        print(f"  {path.name:<28}{counts[path]:>12,} records", flush=True)
    return counts


def write_sample(paths: list, budget: int, destination) -> int:
    """Write `budget` records, split across corpora in proportion to their size.

    Every corpus contributes at its own stride rather than its first N lines,
    so a sample never becomes a prefix of one file.
    """

    counts = line_counts(paths)
    total = sum(counts.values())
    written = 0

    with destination.open("w", encoding="utf-8", newline="\n") as writer:
        for path, count in counts.items():
            share = max(1, round(budget * count / total))
            stride = max(1, count // share)
            taken = 0

            with path.open(encoding="utf-8", errors="replace") as reader:
                for index, line in enumerate(reader):
                    if index % stride:
                        continue
                    text = prepare_text_for_tokenizer(line.strip())
                    if not text:
                        continue
                    writer.write(text + "\n")
                    taken += 1
                    if taken >= share:
                        break

            written += taken
            print(f"  {path.name:<28}{taken:>12,} sampled  (every {stride:,}th record)", flush=True)

    return written


def main() -> None:
    config.TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = config.TOKENIZER_DIR / SAMPLE_FILE

    print("Training the SentencePiece tokenizer...", flush=True)
    print(f"vocab_size={config.TOKENIZER_VOCAB_SIZE:,}  model_type={config.TOKENIZER_MODEL_TYPE}", flush=True)
    print(f"character_coverage={config.TOKENIZER_CHARACTER_COVERAGE}", flush=True)

    paths = iter_data_files()
    start = time.perf_counter()
    written = write_sample(paths, config.TOKENIZER_MAX_SENTENCES, sample_path)
    print(f"sampled {written:,} records into {sample_path.name} in {time.perf_counter() - start:.0f}s", flush=True)

    start = time.perf_counter()
    spm.SentencePieceTrainer.train(
        input=str(sample_path),
        model_prefix=str(config.TOKENIZER_MODEL_PATH.with_suffix("")),
        vocab_size=config.TOKENIZER_VOCAB_SIZE,
        model_type=config.TOKENIZER_MODEL_TYPE,
        character_coverage=config.TOKENIZER_CHARACTER_COVERAGE,
        byte_fallback=config.TOKENIZER_BYTE_FALLBACK,
        max_sentence_length=config.TOKENIZER_MAX_SENTENCE_LENGTH,
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
    print(f"trained in {time.perf_counter() - start:.0f}s", flush=True)

    sample_path.unlink(missing_ok=True)
    print(f"Saved: {config.TOKENIZER_MODEL_PATH}", flush=True)
    print(f"Saved: {config.TOKENIZER_VOCAB_PATH}", flush=True)


if __name__ == "__main__":
    main()
