"""Choose a tokenizer algorithm and vocabulary size by measuring, not by lore.

The v1.1 tokenizer was trained on children's stories alone and packs 2.98
characters into a token there. On the encyclopaedic text that now makes up most
of the corpus it manages 1.20, and 10% of its tokens are byte fallbacks: raw
UTF-8 fragments the model has no way to learn from. Retraining is settled; what
is not settled is which algorithm and how large a vocabulary.

Both questions trade against each other and against the parameter budget. The
embedding is vocab_size x d_model, so at d_model 384 a 32,000-entry vocabulary
costs 12.3M parameters, most of a 17M model, while a smaller vocabulary spends
those parameters on layers instead and pays for it in sequence length.

This trains candidates on a sample and scores them on held-out text from every
domain, reporting the two numbers that matter: characters per token, which sets
how much text fits in the context window and how many tokens a training budget
buys, and the byte-fallback rate, which counts tokens carrying no meaning at
all.
"""

from __future__ import annotations

import argparse
import json
import time
import unicodedata
from pathlib import Path

import sentencepiece as spm

import config

# Held-out lines per corpus, taken from the end so training samples never
# overlap them.
EVAL_LINES = 3_000


def refuse_growing_files(paths: list[Path], settle_seconds: float = 2.0) -> None:
    """Refuse a corpus that another process is still writing.

    Reading a file mid-write produced a tokenizer comparison whose held-out set
    for the largest corpus was empty, so the table was scored on three corpora
    while claiming four. The failure is silent: the numbers look plausible.
    """

    sizes = {path: path.stat().st_size for path in paths}
    time.sleep(settle_seconds)

    growing = [path.name for path in paths if path.stat().st_size != sizes[path]]
    if growing:
        raise RuntimeError(
            f"Still being written: {', '.join(growing)}. Wait for the producing job to finish; "
            "sampling a partial corpus silently biases every number here."
        )


def corpus_files(data_dir: Path) -> list[Path]:
    """Prefer decontaminated corpora, and never mix a corpus with its own clean copy."""

    cleaned = sorted(data_dir.glob("*.clean.txt"))
    cleaned_stems = {path.name.replace(".clean.txt", "") for path in cleaned}
    plain = [
        path
        for path in sorted(data_dir.glob("*.txt"))
        if not path.name.endswith(".clean.txt") and path.stem not in cleaned_stems
    ]
    return cleaned + plain


def sample_corpus(files: list[Path], per_file: int, skip_tail: int) -> tuple[list[str], dict[str, list[str]]]:
    """Draw training lines from the head and evaluation lines from the tail."""

    training: list[str] = []
    evaluation: dict[str, list[str]] = {}

    for path in files:
        head: list[str] = []
        tail: list[str] = []
        with path.open(encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                text = unicodedata.normalize(config.TEXT_UNICODE_NORMALIZATION, line.strip())
                if not text:
                    continue
                if index < per_file:
                    head.append(text)
                else:
                    tail.append(text)
                    if len(tail) > skip_tail:
                        tail.pop(0)
        training.extend(head)
        evaluation[path.name] = tail[-EVAL_LINES:]
        if not evaluation[path.name]:
            raise ValueError(
                f"{path.name} yielded no held-out lines. A corpus that cannot be scored must not "
                "be silently averaged away."
            )
        print(f"  {path.name:<28} {len(head):>8,} train, {len(evaluation[path.name]):>6,} held out", flush=True)

    return training, evaluation


def train_candidate(lines: list[str], model_type: str, vocab_size: int, workdir: Path) -> Path:
    prefix = workdir / f"{model_type}_{vocab_size}"
    corpus = workdir / "sample.txt"
    if not corpus.exists():
        corpus.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    start = time.perf_counter()
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=config.TOKENIZER_CHARACTER_COVERAGE,
        byte_fallback=config.TOKENIZER_BYTE_FALLBACK,
        pad_id=config.PAD_ID,
        unk_id=config.UNK_ID,
        bos_id=config.BOS_ID,
        eos_id=config.EOS_ID,
        num_threads=config.TOKENIZER_NUM_THREADS,
        train_extremely_large_corpus=False,
    )
    print(f"  trained {model_type} {vocab_size:,} in {time.perf_counter() - start:.0f}s", flush=True)
    return prefix.with_suffix(".model")


def score(model_path: Path, evaluation: dict[str, list[str]], d_model: int) -> dict:
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    fallback_ids = {i for i in range(processor.vocab_size()) if processor.id_to_piece(i).startswith("<0x")}

    per_corpus: dict[str, float] = {}
    total_chars = total_tokens = total_fallback = 0

    for name, lines in evaluation.items():
        encoded = processor.encode(lines, out_type=int)
        tokens = sum(len(row) for row in encoded)
        chars = sum(len(line) for line in lines)
        per_corpus[name] = chars / tokens if tokens else 0.0
        total_chars += chars
        total_tokens += tokens
        total_fallback += sum(1 for row in encoded for i in row if i in fallback_ids)

    return {
        "vocab_size": processor.vocab_size(),
        "chars_per_token": total_chars / total_tokens,
        "byte_fallback_rate": total_fallback / total_tokens,
        "embedding_parameters": processor.vocab_size() * d_model,
        "per_corpus_chars_per_token": per_corpus,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=[12_000, 24_000, 32_000])
    parser.add_argument("--model-types", nargs="+", default=["unigram", "bpe"])
    parser.add_argument("--lines-per-corpus", type=int, default=250_000)
    parser.add_argument("--d-model", type=int, default=config.D_MODEL)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--workdir", type=Path, default=Path("runs") / "tokenizer_compare")
    parser.add_argument("--output", type=Path, default=Path("results") / "tokenizer_comparison.json")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.workdir.mkdir(parents=True, exist_ok=True)

    files = corpus_files(args.data_dir)
    if not files:
        raise FileNotFoundError(f"No corpora in {args.data_dir}")

    refuse_growing_files(files)
    print(f"Sampling {len(files)} corpora", flush=True)
    training, evaluation = sample_corpus(files, args.lines_per_corpus, EVAL_LINES * 2)
    print(f"{len(training):,} training lines\n", flush=True)

    results = []
    for model_type in args.model_types:
        for vocab_size in args.vocab_sizes:
            model_path = train_candidate(training, model_type, vocab_size, args.workdir)
            row = {"model_type": model_type, **score(model_path, evaluation, args.d_model)}
            results.append(row)

    # The shipped tokenizer is the thing every candidate has to beat.
    results.insert(
        0,
        {
            "model_type": "current (unigram, stories only)",
            **score(config.TOKENIZER_MODEL_PATH, evaluation, args.d_model),
        },
    )

    header = f"{'tokenizer':<34}{'vocab':>8}{'chars/tok':>11}{'fallback':>10}{'embed params':>14}"
    print("\n" + "=" * len(header), flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for row in results:
        print(
            f"{row['model_type']:<34}{row['vocab_size']:>8,}{row['chars_per_token']:>11.3f}"
            f"{row['byte_fallback_rate']:>9.2%}{row['embedding_parameters']:>14,}",
            flush=True,
        )
    print("=" * len(header), flush=True)
    print("Higher chars/token is better: it fits more text in the context and buys more", flush=True)
    print("text per training token. Byte fallbacks carry no meaning and should be near zero.", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"d_model": args.d_model, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
