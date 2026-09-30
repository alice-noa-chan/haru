"""Document-disjoint Haru v3 corpus, reserved chat tokens, and ground-truth tasks.

No language model is called. Existing corpora are inputs; all new answers are
literal extracts or consequences of the program's explicit state transitions.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import random
import re
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import sentencepiece as spm

from data_utils import blake2b_file, prepare_text_for_tokenizer

WEIGHTS = {"textbooks": 0.40, "web": 0.25, "wiki": 0.15, "story": 0.15, "rules": 0.05}
STORY_WEIGHTS = {"textbooks": 0.25, "web": 0.15, "wiki": 0.05, "story": 0.50, "rules": 0.05}
STORY_CHAT_WEIGHTS = {"grounded_extract": 0.15, "rules": 0.10, "json_format": 0.05, "story": 0.70}
SOURCES = {
    "textbooks": "textbooks.clean.txt",
    "web": "webtext.clean.txt",
    "wiki": "wikipedia.clean.txt",
    "story": "data.txt",
}
ROLES = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
CHAT_TEMPLATE = "{% for message in messages %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
NAMES = {
    "train": [
        "하린",
        "수아",
        "민서",
        "지우",
        "서준",
        "도윤",
        "예린",
        "시우",
        "유나",
        "은우",
        "다은",
        "지호",
        "소율",
        "준서",
        "서연",
        "하준",
        "채원",
        "지안",
        "예준",
        "현우",
        "윤서",
        "태오",
        "하윤",
        "나은",
    ],
    "val": ["보라", "아람", "단비", "태린", "가온", "라온", "이든", "해솔", "초롱", "누리"],
    "test": ["여울", "슬기", "다솜", "한결", "찬솔", "고운", "새봄", "미르"],
}
OBJECTS = {
    "train": ["공", "열쇠", "책", "인형", "연필", "사과", "컵", "종이"],
    "val": ["구슬", "수첩", "모자"],
    "test": ["장갑", "주사위", "배지"],
}
PREFIXES = {
    "train": ["다음 기록을 읽으세요. ", "사건은 아래와 같습니다. ", "", "기록: "],
    "val": ["이 상황에서 답을 찾아 주세요. ", "다음 사실만 고려하세요. "],
    "test": ["새로운 상황을 살펴봅시다. ", "아래 사건을 바탕으로 답하세요. "],
}
LOCATIONS = {"train": ["상자", "서랍", "가방", "선반"], "val": ["바구니", "책장"], "test": ["보관함", "창고"]}


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_for(key):
    value = int(key[:8], 16) % 1000
    return "test" if value < 5 else "val" if value < 10 else "train"


def particle(word, consonant, vowel):
    code = ord(word[-1])
    has_coda = 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0
    return word + (consonant if has_coda else vowel)


def rule_examples(count, split="train", seed=1337, names_split=None, template_split=None):
    from haru.rules import examples

    yield from examples(count, split, seed, names_split, template_split)


def clean(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<[^>]{0,300}>", " ", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r"([^\w\s])\1{5,}", r"\1\1\1", text)
    return text


def canonical(text):
    # Preserve mathematical operators: 2+2 and 2-2 must never be deduplicated.
    return re.sub(r"[\s.,!?\"'…。:;()\[\]{}]+", "", text.casefold())


def probability_mix(weights, names):
    if set(weights) != set(names) or any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Mixture weights must be finite, nonnegative, and include every domain")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Mixture weights must sum to one")
    return [weights[name] for name in names]


def data_profile(root):
    manifest = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
    weights = manifest.get("weights", WEIGHTS)
    probability_mix(weights, WEIGHTS)
    chat_weights = manifest.get(
        "chat_weights", {"grounded_extract": 0.45, "rules": 0.35, "json_format": 0.10, "story": 0.10}
    )
    probability_mix(chat_weights, STORY_CHAT_WEIGHTS)
    return manifest, weights, chat_weights


def deduplicate(source_root, output):
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(output / "dedup.sqlite3")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY)")
    db.execute("CREATE TABLE IF NOT EXISTS near (prefix TEXT PRIMARY KEY, text TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS completed (domain TEXT PRIMARY KEY, stats TEXT)")
    db.commit()
    manifest = {"weights": WEIGHTS, "sources": {}, "split_rule": "sha256 normalized document; 99/0.5/0.5%"}
    for domain, filename in SOURCES.items():
        source = source_root / filename
        marker = output / f"{domain}.complete.json"
        completed = db.execute("SELECT stats FROM completed WHERE domain=?", (domain,)).fetchone()
        if completed:
            stats = json.loads(completed[0])
            if source.stat().st_size != stats["bytes"] or source.stat().st_mtime_ns != stats["mtime_ns"]:
                raise ValueError(f"Completed input changed: {source}")
            if any(not (output / f"{domain}.{split}.jsonl").exists() for split in ("train", "val", "test")):
                raise ValueError(f"Completed corpus output is missing: {domain}")
            marker.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest["sources"][domain] = stats
            continue
        if not source.exists():
            raise FileNotFoundError(source)
        handles = {
            split: (output / f"{domain}.{split}.jsonl").open("w", encoding="utf-8", newline="\n")
            for split in ("train", "val", "test")
        }
        stats = {
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
            "kept": 0,
            "exact_duplicates": 0,
            "near_duplicates": 0,
            "empty": 0,
        }
        # A source transaction rolls back on interruption. Completed earlier sources remain committed.
        try:
            with source.open(encoding="utf-8") as reader:
                for line in reader:
                    text = clean(line)
                    normalized = canonical(text)
                    if len(normalized) < 10:
                        stats["empty"] += 1
                        continue
                    key = digest(normalized)
                    if db.execute("SELECT 1 FROM seen WHERE hash=?", (key,)).fetchone():
                        stats["exact_duplicates"] += 1
                        continue
                    prefix = digest(normalized[:64])
                    neighbor = db.execute("SELECT text FROM near WHERE prefix=?", (prefix,)).fetchone()
                    if neighbor and abs(len(normalized) - len(neighbor[0])) <= 0.05 * max(
                        len(normalized), len(neighbor[0])
                    ):
                        matcher = difflib.SequenceMatcher(None, normalized, neighbor[0], autojunk=True)
                        if matcher.quick_ratio() >= 0.95 and matcher.ratio() >= 0.95:
                            stats["near_duplicates"] += 1
                            db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (key,))
                            continue
                    db.execute("INSERT INTO seen VALUES (?)", (key,))
                    db.execute("INSERT OR IGNORE INTO near VALUES (?,?)", (prefix, normalized))
                    split = split_for(prefix)  # Related-prefix clusters cannot straddle splits.
                    handles[split].write(
                        json.dumps(
                            {"text": text, "source": domain, "split": split, "document_id": key}, ensure_ascii=False
                        )
                        + "\n"
                    )
                    stats["kept"] += 1
                    if stats["kept"] % 100_000 == 0:
                        print(domain, stats, flush=True)
            for handle in handles.values():
                handle.close()
            stats["blake2b16"] = blake2b_file(source)
            db.execute("INSERT INTO completed VALUES (?,?)", (domain, json.dumps(stats, ensure_ascii=False)))
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            for handle in handles.values():
                handle.close()
        marker.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["sources"][domain] = stats
    db.close()
    for split, count in (("train", 100_000), ("val", 4000), ("test", 4000)):
        path = output / f"rules.{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as writer:
            for item in rule_examples(count, split, seed=1337 + (split != "train") + (split == "test")):
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")
    (output / "corpus.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def train_tokenizer(output, sample_docs=2_000_000, sample_characters=50_000_000):
    sample = output / "tokenizer_sample.txt"
    segment_length = 1000
    sampling = {}
    with sample.open("w", encoding="utf-8", newline="\n") as writer:
        for domain, weight in WEIGHTS.items():
            path = output / f"{domain}.train.jsonl"
            share = max(1, round(min(sample_docs, sample_characters // segment_length) * weight))
            reservoir = []
            rng = random.Random(1729)
            count = 0
            pending = ""
            with path.open(encoding="utf-8") as reader:
                for line in reader:
                    pending += json.loads(line)["text"] + "\n"
                    consumed = len(pending) // segment_length * segment_length
                    for start in range(0, consumed, segment_length):
                        piece = pending[start : start + segment_length]
                        if len(reservoir) < share:
                            reservoir.append(piece)
                        else:
                            slot = rng.randrange(count + 1)
                            if slot < share:
                                reservoir[slot] = piece
                        count += 1
                    pending = pending[consumed:]
            sampling[domain] = {
                "segments_seen": count,
                "sample_segments": len(reservoir),
                "characters": sum(map(len, reservoir)),
            }
            for text in reservoir:
                writer.write(prepare_text_for_tokenizer(text) + "\n")
            print("tokenizer sample", domain, sampling[domain], flush=True)
    (output / "tokenizer_sampling.json").write_text(
        json.dumps(
            {"seed": 1729, "max_characters": sample_characters, "segment_length": segment_length, "domains": sampling},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    spm.SentencePieceTrainer.train(
        input=str(sample),
        model_prefix=str(output / "tokenizer"),
        vocab_size=12_000,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        hard_vocab_limit=True,
        max_sentence_length=16384,
        num_threads=16,
        shuffle_input_sentence=True,
        input_sentence_size=sample_docs,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["<|nl|>", "<|literal_nl|>", *ROLES],
    )
    sample.unlink()


def chat_record(item, domain, split):
    text = item["text"]
    if domain == "rules":
        return item
    if domain == "story" and len(text) > 120:
        middle = len(text) // 2
        return {
            "messages": [
                {"role": "user", "content": "다음 이야기를 이어 써 주세요.\n" + text[:middle]},
                {"role": "assistant", "content": text[middle:]},
            ],
            "task": "story",
            "source": domain,
            "split": split,
        }
    first = re.split(r"(?<=[.!?。])\s+", text, maxsplit=1)[0]
    return {
        "messages": [
            {"role": "user", "content": "다음 글의 첫 번째 문장을 그대로 인용하세요.\n글: " + text},
            {"role": "assistant", "content": first},
        ],
        "task": "grounded_extract",
        "source": domain,
        "split": split,
    }


def pack(output):
    sp = spm.SentencePieceProcessor(model_file=str(output / "tokenizer.model"))
    manifest = json.loads((output / "corpus.json").read_text(encoding="utf-8"))
    manifest["tokenizer_blake2b16"] = blake2b_file(output / "tokenizer.model")
    manifest["streams"] = {}
    for split in ("train", "val", "test"):
        chat_path = output / f"chat.{split}.jsonl"
        with chat_path.open("w", encoding="utf-8", newline="\n") as chat:
            for domain in WEIGHTS:
                source = output / f"{domain}.{split}.jsonl"
                path = output / f"{domain}.{split}.bin"
                temporary = path.with_suffix(".bin.tmp")
                total = 0
                chat_samples = []
                chat_rng = random.Random(1337)
                with source.open(encoding="utf-8") as reader, temporary.open("wb") as writer:
                    batch = []
                    for index, line in enumerate(reader):
                        item = json.loads(line)
                        batch.append(prepare_text_for_tokenizer(item["text"]))
                        if domain != "rules":
                            if index < 12_000:
                                chat_samples.append(item)
                            else:
                                slot = chat_rng.randrange(index + 1)
                                if slot < 12_000:
                                    chat_samples[slot] = item
                        if domain == "rules":
                            chat.write(json.dumps(chat_record(item, domain, split), ensure_ascii=False) + "\n")
                            if domain == "rules" and index % 4 == 0:
                                chat.write(
                                    json.dumps(
                                        {
                                            "messages": [
                                                {
                                                    "role": "user",
                                                    "content": "JSON 형식으로 answer 키에 답하세요.\n" + item["prompt"],
                                                },
                                                {
                                                    "role": "assistant",
                                                    "content": json.dumps(
                                                        {"answer": item["answer"]}, ensure_ascii=False
                                                    ),
                                                },
                                            ],
                                            "source": "program_rules",
                                            "task": "json_format",
                                            "split": split,
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                        if len(batch) == 512:
                            encoded = sp.encode(batch, out_type=int, num_threads=16)
                            flat = [token for ids in encoded for token in [*ids, sp.eos_id()]]
                            np.asarray(flat, dtype=np.uint16).tofile(writer)
                            total += len(flat)
                            batch.clear()
                    if batch:
                        encoded = sp.encode(batch, out_type=int, num_threads=16)
                        flat = [token for ids in encoded for token in [*ids, sp.eos_id()]]
                        np.asarray(flat, dtype=np.uint16).tofile(writer)
                        total += len(flat)
                temporary.replace(path)
                for item in chat_samples:
                    chat.write(json.dumps(chat_record(item, domain, split), ensure_ascii=False) + "\n")
                manifest["streams"][f"{domain}.{split}"] = {"tokens": total, "blake2b16": blake2b_file(path)}
                print("packed", domain, split, total, flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


class MixtureSampler:
    def __init__(self, root, seed=1337, split="train"):
        self.root = Path(root)
        self.rng = np.random.default_rng(seed)
        self.domains = list(WEIGHTS)
        _, weights, _ = data_profile(root)
        self.probabilities = probability_mix(weights, self.domains)
        self.streams = {
            domain: np.memmap(self.root / f"{domain}.{split}.bin", dtype=np.uint16, mode="r") for domain in self.domains
        }

    def state_dict(self):
        return self.rng.bit_generator.state

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state

    def close(self):
        for stream in self.streams.values():
            stream._mmap.close()
        self.streams.clear()

    def batch(self, size, length):
        rows = []
        for _ in range(size):
            # Draw one complete window at a time so microbatch grouping cannot
            # change the common data order across architecture candidates.
            domain = self.rng.choice(self.domains, p=self.probabilities)
            stream = self.streams[domain]
            if len(stream) <= length:
                raise ValueError(f"{domain} stream too short")
            start = self.rng.integers(0, len(stream) - length)
            rows.append(np.array(stream[start : start + length + 1], dtype=np.int64))
        array = np.stack(rows)
        return array[:, :-1], array[:, 1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("packed/haru-v3"))
    parser.add_argument("--stage", choices=["clean", "tokenizer", "pack", "all"], default="all")
    parser.add_argument("--sample-docs", type=int, default=2_000_000)
    parser.add_argument("--tokenizer-characters", type=int, default=50_000_000)
    args = parser.parse_args()
    if args.stage in ("clean", "all"):
        deduplicate(args.source_root, args.output)
    if args.stage in ("tokenizer", "all"):
        train_tokenizer(args.output, args.sample_docs, args.tokenizer_characters)
    if args.stage in ("pack", "all"):
        pack(args.output)


if __name__ == "__main__":
    main()
