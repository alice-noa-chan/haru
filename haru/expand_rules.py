"""Version a larger unique rule stream while preserving the candidate dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import sentencepiece as spm

from data_utils import blake2b_file, prepare_text_for_tokenizer
from haru.data import canonical, rule_examples
from haru.runtime import atomic_json

TASKS = ("location", "state", "ownership", "transfer", "speaker", "negation", "arithmetic")


def expand(base, output, count=1_000_000):
    base, output = Path(base), Path(output)
    if count < len(TASKS) or base.resolve() == output.resolve():
        raise ValueError("Use a separate output and at least seven examples")
    base_hash = blake2b_file(base / "manifest.json")
    output.mkdir(parents=True, exist_ok=True)
    previous = output / "rule_generation.json"
    if previous.exists():
        record = json.loads(previous.read_text(encoding="utf-8"))
        if (
            record["base_manifest_blake2b16"] != base_hash
            or record["unique_examples"] != count
            or record["tokenizer_blake2b16"] != blake2b_file(base / "tokenizer.model")
            or record["rules.train"]["blake2b16"] != blake2b_file(output / "rules.train.bin")
        ):
            raise ValueError("Existing rule overlay is a different or damaged version; use a new output")
        return record
    sp = spm.SentencePieceProcessor(model_file=str(base / "tokenizer.model"))
    quota = {task: count // len(TASKS) + (i < count % len(TASKS)) for i, task in enumerate(TASKS)}
    accepted, seen = Counter(), set()
    path = output / "rules.train.bin"
    temporary = path.with_name(path.name + ".tmp")
    total, attempts, batch = 0, 0, []

    def write_batch(writer):
        encoded = sp.encode(batch, out_type=int, num_threads=8)
        flat = [token for ids in encoded for token in [*ids, sp.eos_id()]]
        np.asarray(flat, dtype=np.uint16).tofile(writer)
        batch.clear()
        return len(flat)

    with temporary.open("wb") as writer:
        for item in rule_examples(count * 30, "train", seed=1337):
            attempts += 1
            task = item["task"]
            if accepted[task] >= quota[task]:
                continue
            key = hashlib.blake2b(canonical(item["text"]).encode("utf-8"), digest_size=16).digest()
            if key in seen:
                continue
            seen.add(key)
            accepted[task] += 1
            batch.append(prepare_text_for_tokenizer(item["text"]))
            if len(batch) == 512:
                total += write_batch(writer)
            if sum(accepted.values()) == count:
                break
        if sum(accepted.values()) != count:
            raise ValueError("The rule generator exhausted its unique combinations; no dataset was published")
        if batch:
            total += write_batch(writer)
        writer.flush()
        os.fsync(writer.fileno())
    temporary.replace(path)
    record = {
        "base_manifest_blake2b16": base_hash,
        "tokenizer_blake2b16": blake2b_file(base / "tokenizer.model"),
        "seed": 1337,
        "unique_examples": count,
        "task_counts": dict(accepted),
        "attempts": attempts,
        "deduplication": "blake2b16 of canonical text; mathematical operators preserved",
        "rules.train": {"tokens": total, "blake2b16": blake2b_file(path)},
        "held_out": "unchanged candidate validation/test documents",
        "chat": "unchanged assistant-only chat dataset",
    }
    atomic_json(output / "rule_generation.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def materialize(base, overlay, output):
    """Create a new immutable training view; the original streams are never written."""
    base, overlay, output = Path(base), Path(overlay), Path(output)
    if output.resolve() in (base.resolve(), overlay.resolve()):
        raise ValueError("The teacher view must be separate from its inputs")
    record = json.loads((overlay / "rule_generation.json").read_text(encoding="utf-8"))
    base_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    base_hash = base_manifest.get("source_manifest_blake2b16", blake2b_file(base / "manifest.json"))
    if record["base_manifest_blake2b16"] != base_hash:
        raise ValueError("Rule overlay belongs to a different candidate dataset")
    if record["tokenizer_blake2b16"] != blake2b_file(base / "tokenizer.model"):
        raise ValueError("Rule overlay tokenizer changed")
    if record["rules.train"]["blake2b16"] != blake2b_file(overlay / "rules.train.bin"):
        raise ValueError("Rule overlay stream hash mismatch")
    manifest = base_manifest
    manifest["streams"]["rules.train"] = record["rules.train"]
    manifest["rule_expansion"] = record
    existing = output / "manifest.json"
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8")) != manifest:
        raise ValueError("An existing teacher data view has a different version")
    output.mkdir(parents=True, exist_ok=True)
    for path in base.iterdir():
        if not path.is_file() or path.name in ("manifest.json", "rules.train.bin"):
            continue
        if not (
            path.suffix in (".bin", ".model", ".vocab")
            or path.name.startswith("chat.")
            or path.name.endswith((".val.jsonl", ".test.jsonl"))
            or path.name in ("corpus.json", "tokenizer_sampling.json")
        ):
            continue
        destination = output / path.name
        if not destination.exists():
            try:
                os.link(path, destination)
            except OSError:
                shutil.copyfile(path, destination)
    target = output / "rules.train.bin"
    if not target.exists() or blake2b_file(target) != record["rules.train"]["blake2b16"]:
        temporary = target.with_name(target.name + ".tmp")
        shutil.copyfile(overlay / "rules.train.bin", temporary)
        temporary.replace(target)
    atomic_json(existing, manifest)
    atomic_json(output / "rule_generation.json", record)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("packed/haru-v3"))
    parser.add_argument("--output", type=Path, default=Path("packed/haru-v3-rules-1m"))
    parser.add_argument("--count", type=int, default=1_000_000)
    parser.add_argument("--materialize", type=Path)
    args = parser.parse_args()
    if args.materialize:
        materialize(args.base, args.output, args.materialize)
    else:
        expand(args.base, args.output, args.count)


if __name__ == "__main__":
    main()
