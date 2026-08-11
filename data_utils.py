from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import config


@dataclass(frozen=True, slots=True)
class TextRecord:
    """One normalized text sample and its source location."""

    text: str
    source: Path
    line_number: int


def normalize_text(text: str) -> str:
    """Apply minimal, loss-conscious normalization before tokenization."""

    # Normalize platform-specific line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # NFC unifies composed/decomposed Hangul without NFKC's aggressive changes.
    text = unicodedata.normalize(config.TEXT_UNICODE_NORMALIZATION, text)
    return text


def prepare_text_for_tokenizer(text: str) -> str:
    """Escape literal control text and preserve newlines for SentencePiece."""

    text = normalize_text(text)
    text = text.replace(config.NEWLINE_TOKEN, config.NEWLINE_ESCAPE_TOKEN)
    return text.replace("\n", config.NEWLINE_TOKEN)


def restore_text_from_tokenizer(text: str) -> str:
    """Restore encoded newlines and escaped literal control text."""

    text = text.replace(config.NEWLINE_TOKEN, "\n")
    return text.replace(config.NEWLINE_ESCAPE_TOKEN, config.NEWLINE_TOKEN)


def iter_data_files(data_dir: Path | None = None) -> list[Path]:
    """Return supported data files recursively in deterministic order."""

    root = data_dir or config.DATA_DIR
    if not root.exists():
        return []

    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".jsonl"}
    ]
    return sorted(files, key=lambda p: p.as_posix())


def iter_text_records(data_dir: Path | None = None) -> Iterator[TextRecord]:
    """Stream normalized records from every supported source file."""

    files = iter_data_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No training data found. Add .txt or .jsonl files under {config.DATA_DIR}"
        )

    for path in files:
        suffix = path.suffix.lower()

        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                # Remove only the newline that separates file records.
                raw_line = raw_line.rstrip("\n\r")

                if suffix == ".txt":
                    text = raw_line
                else:
                    if not raw_line.strip():
                        continue
                    try:
                        row = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSONL at {path}:{line_number}: {exc}"
                        ) from exc

                    if not isinstance(row, dict):
                        raise ValueError(
                            f"JSONL record must be an object: {path}:{line_number}"
                        )
                    if config.JSONL_TEXT_KEY not in row:
                        raise KeyError(
                            f"JSONL record is missing {config.JSONL_TEXT_KEY!r}: "
                            f"{path}:{line_number}"
                        )
                    text_value = row[config.JSONL_TEXT_KEY]
                    if not isinstance(text_value, str):
                        raise TypeError(
                            f"JSONL field {config.JSONL_TEXT_KEY!r} must be a string: "
                            f"{path}:{line_number}"
                        )
                    text = text_value

                text = normalize_text(text)
                if config.SKIP_EMPTY_TEXT and not text.strip():
                    continue

                yield TextRecord(text=text, source=path, line_number=line_number)


def stable_text_hash(text: str) -> bytes:
    """Return a stable 128-bit hash derived only from normalized text."""

    normalized = normalize_text(text).encode("utf-8")
    return hashlib.blake2b(normalized, digest_size=16).digest()


def is_validation_text(text: str) -> bool:
    """Assign identical text to the same split regardless of file location."""

    digest = stable_text_hash(text)
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    threshold = int(config.VAL_FRACTION * (1 << 64))
    return value < threshold


def file_fingerprint(path: Path) -> dict[str, int | str]:
    """Return a fast metadata fingerprint for a source file."""

    stat = path.stat()
    return {
        "path": str(path.relative_to(config.ROOT_DIR)),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def dataset_fingerprint() -> list[dict[str, int | str]]:
    """Fingerprint the current set of source files."""

    return [file_fingerprint(path) for path in iter_data_files()]


def blake2b_file(path: Path, digest_size: int = 16) -> str:
    """Hash the full contents of a relatively small artifact such as a tokenizer."""

    digest = hashlib.blake2b(digest_size=digest_size)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
