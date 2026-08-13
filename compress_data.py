"""Compress corpora for upload, and decompress them on a training host.

Uploading 2.7 GB of Korean text to a GPU host wastes both transfer time and,
for a pod that bills by the second, money spent waiting. Text compresses well
enough that this is nearly free.

Measured on a 157 MB sample of the real corpus:

    method      size MB   ratio   compress s   decompress s   MB/s decompressed
    gzip -6        41.9    3.75          6.2            0.5                 309
    zstd -3        38.3    4.10          0.3            0.2                 864
    zstd -10       29.9    5.26          1.5            0.2                 960
    zstd -19       21.4    7.37         27.3            0.1               1,098
    xz -1          37.3    4.21          6.3            1.7                  92

zstd at level 19 is the choice: it compresses hardest and still decompresses
fastest, so the slow part happens once here rather than on every host that
unpacks it. xz reaches a similar ratio at level 1 but decompresses twelve times
slower, which is the operation that runs on the machine being paid for.

Originals are never removed. A corpus is expensive to rebuild and the archive
is a copy, not a replacement.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import zstandard as zstd

DEFAULT_LEVEL = 19
READ_CHUNK = 16 * 1024 * 1024


def compress(source: Path, level: int) -> Path:
    destination = source.with_suffix(source.suffix + ".zst")
    compressor = zstd.ZstdCompressor(level=level, threads=-1)

    start = time.perf_counter()
    with source.open("rb") as reader, destination.open("wb") as writer:
        compressor.copy_stream(reader, writer, read_size=READ_CHUNK)
    elapsed = time.perf_counter() - start

    original = source.stat().st_size
    packed = destination.stat().st_size
    print(
        f"  {source.name} {original / 1e9:.2f} GB -> {destination.name} {packed / 1e9:.2f} GB "
        f"({original / packed:.2f}x, {elapsed:.0f}s)",
        flush=True,
    )
    return destination


def decompress(source: Path) -> Path:
    if source.suffix != ".zst":
        raise ValueError(f"{source} is not a .zst archive")

    destination = source.with_suffix("")
    decompressor = zstd.ZstdDecompressor()

    start = time.perf_counter()
    with source.open("rb") as reader, destination.open("wb") as writer:
        decompressor.copy_stream(reader, writer, read_size=READ_CHUNK)
    elapsed = time.perf_counter() - start

    size = destination.stat().st_size
    print(f"  {source.name} -> {destination.name} {size / 1e9:.2f} GB ({size / 1e6 / max(elapsed, 1e-9):.0f} MB/s)")
    return destination


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("compress", "decompress"))
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL, help="zstd level, 1 to 22")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    total_in = total_out = 0

    for path in args.paths:
        if not path.exists():
            raise FileNotFoundError(f"No file at {path}")
        total_in += path.stat().st_size
        result = compress(path, args.level) if args.action == "compress" else decompress(path)
        total_out += result.stat().st_size

    if args.action == "compress":
        print(f"\n{total_in / 1e9:.2f} GB -> {total_out / 1e9:.2f} GB to upload ({total_in / total_out:.2f}x)")
        print("Originals kept. On the host: pip install zstandard && python compress_data.py decompress <file>.zst")
    else:
        print(f"\nrestored {total_out / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
