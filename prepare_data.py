from __future__ import annotations

import json

import numpy as np

import config
from data_utils import blake2b_file, dataset_fingerprint, is_validation_text, iter_text_records
from tokenizer_utils import StoryTokenizer

# Buffer tokens per split to avoid many tiny writes.
WRITE_BUFFER_TOKENS = 1_000_000


def flush_buffer(handle, buffer: list[int]) -> int:
    """Write a uint16 token buffer and return the number of tokens written."""

    if not buffer:
        return 0

    array = np.asarray(buffer, dtype=np.uint16)
    handle.write(array.tobytes(order="C"))
    count = int(array.size)
    buffer.clear()
    return count


def main() -> None:
    tokenizer = StoryTokenizer()

    if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("vocab_size exceeds uint16; choose a wider packing dtype")

    config.PACKED_DIR.mkdir(parents=True, exist_ok=True)

    # Write temporary artifacts first so an interrupted pack keeps old files usable.
    train_tmp = config.TRAIN_BIN_PATH.with_suffix(".bin.tmp")
    val_tmp = config.VAL_BIN_PATH.with_suffix(".bin.tmp")

    train_buffer: list[int] = []
    val_buffer: list[int] = []

    train_records = 0
    val_records = 0
    train_tokens = 0
    val_tokens = 0

    print("Packing source text into token streams...", flush=True)

    with train_tmp.open("wb") as train_handle, val_tmp.open("wb") as val_handle:
        for record_index, record in enumerate(iter_text_records(), start=1):
            ids = tokenizer.encode(record.text, add_bos=True, add_eos=True)

            if is_validation_text(record.text):
                val_records += 1
                val_buffer.extend(ids)
                if len(val_buffer) >= WRITE_BUFFER_TOKENS:
                    val_tokens += flush_buffer(val_handle, val_buffer)
            else:
                train_records += 1
                train_buffer.extend(ids)
                if len(train_buffer) >= WRITE_BUFFER_TOKENS:
                    train_tokens += flush_buffer(train_handle, train_buffer)

            if record_index % 100_000 == 0:
                print(
                    f"records={record_index:,} train={train_records:,} val={val_records:,}",
                    flush=True,
                )

        train_tokens += flush_buffer(train_handle, train_buffer)
        val_tokens += flush_buffer(val_handle, val_buffer)

    train_tmp.replace(config.TRAIN_BIN_PATH)
    val_tmp.replace(config.VAL_BIN_PATH)

    meta = {
        "format": "uint16-packed-token-stream",
        "vocab_size": tokenizer.vocab_size,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "train_records": train_records,
        "val_records": val_records,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "val_fraction": config.VAL_FRACTION,
        "data_files": dataset_fingerprint(),
        "tokenizer": {
            "path": str(config.TOKENIZER_MODEL_PATH.relative_to(config.ROOT_DIR)),
            "bytes": config.TOKENIZER_MODEL_PATH.stat().st_size,
            "blake2b16": blake2b_file(config.TOKENIZER_MODEL_PATH),
        },
    }

    meta_tmp = config.PACKED_META_PATH.with_suffix(".json.tmp")
    meta_tmp.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta_tmp.replace(config.PACKED_META_PATH)

    print("Packing complete", flush=True)
    print(f"train_records={train_records:,}", flush=True)
    print(f"val_records={val_records:,}", flush=True)
    print(f"train_tokens={train_tokens:,}", flush=True)
    print(f"val_tokens={val_tokens:,}", flush=True)
    print(f"meta={config.PACKED_META_PATH}", flush=True)


if __name__ == "__main__":
    main()
