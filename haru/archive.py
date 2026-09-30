"""Hashed zstd bundles for moving prepared data between local machines."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

from haru.runtime import atomic_json


def file_hash(path):
    digest = hashlib.blake2b(digest_size=16)
    with Path(path).open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_paths(root, training_data=False):
    root = Path(root).resolve()
    paths = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    if training_data:
        # Training JSONL and the dedup database are unnecessary after tokenization.
        paths = [
            p
            for p in paths
            if p.suffix == ".bin"
            or p.name
            in ("tokenizer.model", "tokenizer.vocab", "manifest.json", "corpus.json", "tokenizer_sampling.json")
            or p.name.startswith("chat.")
            or p.name.endswith((".val.jsonl", ".test.jsonl"))
        ]
        if not (root / "manifest.json").exists():
            raise FileNotFoundError("Tokenization must finish before bundling data")
    if not paths:
        raise ValueError("Empty bundle")
    if any(p.is_symlink() or root not in p.resolve().parents for p in paths):
        raise ValueError("Bundle sources must be regular files within the selected directory")
    return paths


def bundle(root, destination, training_data=False):
    import zstandard as zstd

    root, destination = Path(root).resolve(), Path(destination).resolve()
    paths = bundle_paths(root, training_data)
    records = {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "blake2b16": file_hash(p)} for p in paths}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as writer, zstd.ZstdCompressor(level=3, threads=2).stream_writer(writer) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for path in paths:
                archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    temporary.replace(destination)
    payload = {"bundle_blake2b16": file_hash(destination), "files": records}
    atomic_json(destination.with_name(destination.name + ".json"), payload)
    print(
        json.dumps({"bundle": str(destination), "bytes": destination.stat().st_size, "files": len(paths)}), flush=True
    )
    return payload


def unpack(source, destination):
    import zstandard as zstd

    source, destination = Path(source), Path(destination)
    manifest = json.loads(source.with_name(source.name + ".json").read_text(encoding="utf-8"))
    if file_hash(source) != manifest["bundle_blake2b16"]:
        raise ValueError("Bundle hash mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    seen = set()
    with source.open("rb") as reader, zstd.ZstdDecompressor().stream_reader(reader) as decompressed:
        with tarfile.open(fileobj=decompressed, mode="r|") as archive:
            for item in archive:
                name = PurePosixPath(item.name)
                if (
                    not item.isfile()
                    or name.is_absolute()
                    or ".." in name.parts
                    or "\\" in item.name
                    or ":" in item.name
                    or item.name not in manifest["files"]
                    or item.name in seen
                ):
                    raise ValueError(f"Unexpected bundle member: {item.name}")
                archive.extract(item, destination, filter="data")
                path = destination.joinpath(*name.parts)
                record = manifest["files"][item.name]
                if path.stat().st_size != record["bytes"] or file_hash(path) != record["blake2b16"]:
                    raise ValueError(f"Extracted file hash mismatch: {item.name}")
                seen.add(item.name)
    if seen != set(manifest["files"]):
        raise ValueError("Incomplete bundle")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("bundle", "unpack"))
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--training-data", action="store_true")
    args = parser.parse_args()
    if args.action == "bundle":
        bundle(args.source, args.destination, args.training_data)
    elif args.action == "unpack":
        unpack(args.source, args.destination)


if __name__ == "__main__":
    main()
