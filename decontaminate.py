"""Remove benchmark text from a training corpus before it is trained on.

Refusing to download KoBEST is not enough. KoBEST is built from ordinary Korean
sources: its BoolQ passages come from encyclopaedic text and SentiNeg from
reviews, so a Wikipedia dump or a web crawl can contain the exact sentences the
benchmark asks about even though the benchmark files were never fetched. A score
from a corpus that was never checked cannot be trusted upward.

The check is n-gram overlap. Every benchmark record is reduced to a set of
character n-grams, and any training line sharing one is dropped.

Character n-grams rather than word n-grams, because Korean is agglutinative and
whitespace-delimited words are unreliable units: the same content word appears
with different particles attached, and a word-level match would miss the
paraphrase-free copies this is meant to catch.

The default length of 40 characters is long enough that a collision is a real
quotation rather than a common phrase. Shorter windows start dropping ordinary
sentences, which quietly shrinks the corpus and biases what remains.

n-grams are stored as 64-bit hashes. Keeping the strings themselves costs about
an order of magnitude more memory for no benefit, since only membership is ever
tested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

# Every split the project evaluates on, plus the splits those are drawn from.
BENCHMARK_SPLITS: dict[str, tuple[str, str | None, tuple[str, ...]]] = {
    "kobest_boolq": ("skt/kobest_v1", "boolq", ("paragraph", "question")),
    "kobest_copa": ("skt/kobest_v1", "copa", ("premise", "alternative_1", "alternative_2")),
    "kobest_hellaswag": ("skt/kobest_v1", "hellaswag", ("context", "ending_1", "ending_2", "ending_3", "ending_4")),
    "kobest_sentineg": ("skt/kobest_v1", "sentineg", ("sentence",)),
    "kobest_wic": ("skt/kobest_v1", "wic", ("context_1", "context_2")),
}

DEFAULT_NGRAM = 40


def normalize(text: str) -> str:
    """Fold away the differences that would let a copy slip past the match."""

    folded = unicodedata.normalize("NFC", text).lower()
    return "".join(character for character in folded if not character.isspace())


def ngrams(text: str, size: int) -> set[int]:
    """Hash every character window of `size` in the normalized text."""

    cleaned = normalize(text)
    if len(cleaned) < size:
        return set()
    return {
        int.from_bytes(hashlib.blake2b(cleaned[i : i + size].encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(len(cleaned) - size + 1)
    }


def build_benchmark_index(tasks: list[str], size: int, splits: tuple[str, ...]) -> set[int]:
    """Collect n-grams from every benchmark record that could be memorized."""

    from datasets import load_dataset

    index: set[int] = set()
    for task in tasks:
        repo, name, fields = BENCHMARK_SPLITS[task]
        collected = 0
        for split in splits:
            try:
                dataset = load_dataset(repo, name, split=split)
            except (ValueError, KeyError):
                continue
            for record in dataset:
                for field in fields:
                    value = record.get(field)
                    if isinstance(value, str) and value.strip():
                        index |= ngrams(value, size)
                        collected += 1
        print(f"  {task:<20} {collected:>7,} fields indexed", flush=True)

    print(f"benchmark index: {len(index):,} distinct {size}-character n-grams", flush=True)
    return index


def filter_corpus(source: Path, destination: Path, index: set[int], size: int) -> dict[str, int]:
    """Copy `source` to `destination`, dropping lines that quote the benchmark."""

    kept = dropped = 0
    with source.open("r", encoding="utf-8", errors="replace") as reader:
        with destination.open("w", encoding="utf-8") as writer:
            for line in reader:
                text = line.strip()
                if not text:
                    continue
                if ngrams(text, size) & index:
                    dropped += 1
                    continue
                writer.write(text + "\n")
                kept += 1
                if (kept + dropped) % 200_000 == 0:
                    print(f"  {kept + dropped:,} lines, {dropped:,} dropped", flush=True)

    return {"kept": kept, "dropped": dropped}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpora", nargs="+", type=Path, help="Corpus files under data/ to clean")
    parser.add_argument("--tasks", nargs="+", default=list(BENCHMARK_SPLITS), choices=list(BENCHMARK_SPLITS))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("test", "validation"),
        help="Benchmark splits to index. Test is what is scored; validation is indexed too because it "
        "is the obvious thing to tune on.",
    )
    parser.add_argument("--ngram", type=int, default=DEFAULT_NGRAM, help="Character window length")
    parser.add_argument("--suffix", default=".clean", help="Written beside each input as <name><suffix>.txt")
    parser.add_argument("--report", type=Path, default=Path("results") / "decontamination.json")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.ngram < 16:
        raise ValueError(f"--ngram {args.ngram} is short enough to match ordinary phrases, not quotations")

    print(f"Indexing {len(args.tasks)} benchmark(s) over splits {args.splits}", flush=True)
    index = build_benchmark_index(args.tasks, args.ngram, tuple(args.splits))

    report: dict[str, dict[str, int]] = {}
    for corpus in args.corpora:
        if not corpus.exists():
            raise FileNotFoundError(f"No corpus at {corpus}")
        destination = corpus.with_name(corpus.stem + args.suffix + corpus.suffix)
        print(f"\n{corpus.name} -> {destination.name}", flush=True)

        counts = filter_corpus(corpus, destination, index, args.ngram)
        total = counts["kept"] + counts["dropped"]
        rate = counts["dropped"] / total if total else 0.0
        counts["drop_rate"] = rate
        report[corpus.name] = counts

        print(f"  kept {counts['kept']:,}, dropped {counts['dropped']:,} ({rate:.4%})", flush=True)
        # A large drop rate means the window is matching ordinary language, not
        # quotations, and the corpus is being silently truncated.
        if rate > 0.01:
            print(f"  WARNING: {rate:.2%} is high for a {args.ngram}-character window. Inspect before training.")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            {"ngram": args.ngram, "tasks": args.tasks, "splits": list(args.splits), "corpora": report},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {args.report}", flush=True)
    print("Record the drop counts in data/datasets.md before reporting any benchmark score.", flush=True)


if __name__ == "__main__":
    main()
