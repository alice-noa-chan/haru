"""Version a story-first training view without changing the prepared corpus."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from data_utils import blake2b_file
from haru.archive import bundle_paths
from haru.data import STORY_CHAT_WEIGHTS, STORY_WEIGHTS, probability_mix
from haru.runtime import atomic_json


def materialize_story_view(base, output):
    base, output = Path(base), Path(output)
    if base.resolve() == output.resolve():
        raise ValueError("The story view must be separate from the original data")
    source_hash = blake2b_file(base / "manifest.json")
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("objective"):
        raise ValueError("Create a story view from the original packed corpus")
    probability_mix(STORY_WEIGHTS, STORY_WEIGHTS)
    probability_mix(STORY_CHAT_WEIGHTS, STORY_CHAT_WEIGHTS)
    if blake2b_file(base / "tokenizer.model") != manifest["tokenizer_blake2b16"]:
        raise ValueError("Tokenizer hash differs from the corpus manifest")
    for name, record in manifest["streams"].items():
        split_domain, split = name.split(".")
        path = base / f"{split_domain}.{split}.bin"
        if blake2b_file(path) != record["blake2b16"]:
            raise ValueError(f"Training stream hash mismatch: {name}")
    manifest.update(
        {
            "objective": "story_continuation",
            "objective_version": 1,
            "source_manifest_blake2b16": source_hash,
            "weights": STORY_WEIGHTS,
            "chat_weights": STORY_CHAT_WEIGHTS,
            "tokenizer_note": "Reuses the original 12K tokenizer; no tokenizer comparison has been measured",
        }
    )
    existing = output / "manifest.json"
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Existing story view has a different manifest")
    output.mkdir(parents=True, exist_ok=True)
    for path in bundle_paths(base, training_data=True):
        if path.name == "manifest.json":
            continue
        destination = output / path.name
        if destination.exists():
            if destination.stat().st_size != path.stat().st_size or blake2b_file(destination) != blake2b_file(path):
                raise ValueError(f"Existing story view file differs: {destination}")
            continue
        try:
            os.link(path, destination)
        except OSError:
            shutil.copyfile(path, destination)
    atomic_json(existing, manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("packed/haru-v3"))
    parser.add_argument("--output", type=Path, default=Path("packed/haru-v3-story"))
    args = parser.parse_args()
    print(materialize_story_view(args.base, args.output))


if __name__ == "__main__":
    main()
