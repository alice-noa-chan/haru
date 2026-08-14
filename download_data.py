"""Download Korean training corpora into data/ as line-delimited text.

The corpora themselves are never committed: `data/*` is gitignored, and
`data/datasets.md` records which sources were used and under what licence.

Two rules this script enforces rather than documents:

- It refuses to write `data/data.txt`, the existing packed corpus.
- It refuses to download any benchmark dataset. KoBEST is the evaluation, so
  training on it would make every score meaningless, and a typo in a dataset id
  should fail loudly rather than quietly contaminate the corpus.

Decontamination against benchmark splits is a separate step and is not done
here. Any corpus built from encyclopaedic or review text can contain the exact
passages KoBEST asks about even when the benchmark files were never fetched.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import config

# Text fields differ per dataset; the first present one is used.
TEXT_FIELDS = ("text", "content", "document", "instruction", "body")

# Which corpora to fetch is read from data/sources.json at run time, never
# hard-coded here. data/* is gitignored, so the list of datasets stays out of
# the repository along with the text itself.
SOURCES_FILE = "sources.json"

EXAMPLE_SOURCES = """{
  "short-name": {
    "repo": "org/dataset",
    "config": null,
    "split": "train",
    "licence": "the licence on the dataset card",
    "note": "why this corpus is worth its gigabytes"
  }
}"""


def load_sources(data_dir: Path) -> dict[str, dict]:
    """Read the corpus list from the gitignored data directory."""

    path = data_dir / SOURCES_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No corpus list at {path}. Create it with entries shaped like:\n\n{EXAMPLE_SOURCES}\n\n"
            "It stays out of Git with the rest of data/."
        )

    sources = json.loads(path.read_text(encoding="utf-8"))
    for name, source in sources.items():
        missing = [key for key in ("repo", "split") if not source.get(key)]
        if missing:
            raise ValueError(f"Source {name!r} in {path} is missing: {', '.join(missing)}")
    return sources


# Substrings that identify an evaluation set. Matching ids are refused.
BENCHMARK_MARKERS = ("kobest", "klue", "haerae", "kmmlu", "nsmc", "korquad", "hellaswag", "boolq", "copa")

PROTECTED_FILES = {"data.txt"}


def refuse_benchmarks(repo: str) -> None:
    """Refuse evaluation sets, matching the dataset name rather than the owner.

    Matching the whole repository path rejected HAERAE-HUB/KOREAN-WEBTEXT, a
    training corpus, because the organisation also publishes HAE-RAE Bench. The
    owner says who released a dataset, not what it is.

    Separators are stripped before matching so HAE_RAE_BENCH and haerae-bench
    both still fail, which the previous whole-path check would have missed for
    the underscored spelling.
    """

    name = repo.rsplit("/", 1)[-1].lower()
    condensed = name.replace("-", "").replace("_", "").replace(".", "")

    for marker in BENCHMARK_MARKERS:
        if marker in condensed:
            raise ValueError(
                f"Refusing to download {repo!r}: its name matches the benchmark marker {marker!r}. "
                "Training on evaluation data would invalidate every score this project reports."
            )


def normalize(text: str) -> str:
    """One record per line, NFC, matching what prepare_data.py expects."""

    cleaned = unicodedata.normalize(config.TEXT_UNICODE_NORMALIZATION, text).strip()
    return " ".join(cleaned.split())


def download(name: str, source: dict, destination: Path, max_records: int | None, force: bool = False) -> int:
    from datasets import load_dataset

    refuse_benchmarks(source["repo"])
    if destination.name in PROTECTED_FILES:
        raise ValueError(f"Refusing to overwrite {destination}, which holds the existing training corpus")

    # A --max-records sample writes to the same path as the full download, so a
    # quick check silently replaced a finished 988 MB corpus with 20 records and
    # truncated the decontamination reading it at the time. Overwriting an
    # existing corpus is now deliberate.
    if destination.exists() and destination.stat().st_size > 0 and not force:
        raise FileExistsError(
            f"{destination} already holds {destination.stat().st_size / 1e6:.0f} MB. "
            "Pass --force to replace it, or --data-dir to write a sample somewhere else."
        )

    print(f"\n{name}: {source['repo']} ({source.get('licence', 'licence unrecorded')})", flush=True)
    if source.get("note"):
        print(f"  {source['note']}", flush=True)

    dataset = load_dataset(source["repo"], source.get("config"), split=source["split"], streaming=True)

    written = 0
    skipped = 0
    # Force LF. Text mode on Windows rewrites every line terminator to CRLF,
    # which adds a byte per line to a corpus destined for a Linux training host
    # and makes the same download differ between machines.
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in dataset:
            field = next((f for f in TEXT_FIELDS if record.get(f)), None)
            if field is None:
                skipped += 1
                continue
            line = normalize(str(record[field]))
            if not line:
                skipped += 1
                continue
            handle.write(line + "\n")
            written += 1
            if written % 50_000 == 0:
                print(f"  {written:,} records", flush=True)
            if max_records is not None and written >= max_records:
                break

    size = destination.stat().st_size
    print(f"  wrote {written:,} records, {size / 1e9:.2f} GB -> {destination}", flush=True)
    if skipped:
        print(f"  skipped {skipped:,} records with no usable text field", flush=True)
    return written


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="*", help=f"Names from data/{SOURCES_FILE}; default is all of them")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Stop each source after N records. Use for a sample before committing disk to the full download.",
    )
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing corpus file. Without it, a sample cannot clobber a finished download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    sources = load_sources(args.data_dir)

    selected = args.sources or list(sources)
    unknown = [name for name in selected if name not in sources]
    if unknown:
        raise ValueError(f"Not in data/{SOURCES_FILE}: {', '.join(unknown)}")

    total = 0
    for name in selected:
        total += download(name, sources[name], args.data_dir / f"{name}.txt", args.max_records, args.force)

    print(f"\n{total:,} records across {len(selected)} source(s) in {args.data_dir}", flush=True)
    print("data/* is gitignored. Record what you used in data/datasets.md.", flush=True)
    print("Decontaminate against the KoBEST splits before reporting a benchmark score.", flush=True)


if __name__ == "__main__":
    main()
