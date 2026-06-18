#!/usr/bin/env python3
"""Prepare GPT-2 token streams from Shakespeare text.

This helper deliberately supports two modes:

1. --inspect: no third-party packages required. It downloads the source text
   and prints a few training chunks.
2. tokenization: requires `tiktoken` and writes train.bin/val.bin compatible with
   the upstream GPT-2-pretraining notebook style.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCE_NAME = "tiny-shakespeare"
SOURCE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass
class ExtractedRecord:
    source: str
    text: str


def http_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    return data.decode("utf-8", "replace")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def is_useful_text(text: str) -> bool:
    text = clean_text(text)
    if len(text) < 20:
        return False
    return True


def extract_records_from_text(text: str, max_records: int | None = None) -> list[ExtractedRecord]:
    paragraphs = re.split(r"\n\s*\n", clean_text(text))
    records: list[ExtractedRecord] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        paragraph = clean_text(paragraph)
        if not is_useful_text(paragraph):
            continue
        if current and current_len + len(paragraph) > 1800:
            idx = len(records)
            records.append(ExtractedRecord(source=f"{SOURCE_NAME}:chunk-{idx}", text="\n\n".join(current)))
            current = []
            current_len = 0
            if max_records is not None and len(records) >= max_records:
                return records[:max_records]
        current.append(paragraph)
        current_len += len(paragraph)
    if current:
        idx = len(records)
        records.append(ExtractedRecord(source=f"{SOURCE_NAME}:chunk-{idx}", text="\n\n".join(current)))
    return records[:max_records] if max_records is not None else records


def collect_records(source_url: str, max_records: int | None) -> list[ExtractedRecord]:
    return extract_records_from_text(http_text(source_url), max_records=max_records)


def write_tokens(records: list[ExtractedRecord], out_dir: Path, val_fraction: float, seed: int, source_url: str) -> dict[str, Any]:
    try:
        import tiktoken  # type: ignore
    except Exception as exc:
        raise SystemExit("Tokenization requires `tiktoken`") from exc

    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    val_n = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    val_records = shuffled[:val_n]
    train_records = shuffled[val_n:]
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    def encode_many(rs: list[ExtractedRecord]) -> array:
        toks = array("H")  # GPT-2 vocab max 50256 fits uint16.
        for rec in rs:
            ids = enc.encode(rec.text, allowed_special={"<|endoftext|>"})
            toks.extend(ids)
            toks.append(eot)
        return toks

    out_dir.mkdir(parents=True, exist_ok=True)
    train = encode_many(train_records)
    val = encode_many(val_records)
    (out_dir / "train.bin").write_bytes(train.tobytes())
    (out_dir / "val.bin").write_bytes(val.tobytes())
    meta = {
        "source": SOURCE_NAME,
        "source_url": source_url,
        "num_records": len(records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "train_tokens": len(train),
        "val_tokens": len(val),
        "dtype": "uint16",
        "tokenizer": "tiktoken:gpt2",
        "seed": seed,
        "val_fraction": val_fraction,
        "sample_sources": [r.source for r in records[:10]],
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--out-dir", type=Path, default=Path("data/shakespeare_tokens"))
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    records = collect_records(source_url=args.source_url, max_records=args.max_records)
    print(f"source={SOURCE_NAME}")
    print(f"source_url={args.source_url}")
    print(f"records={len(records)}")
    for rec in records[:5]:
        preview = rec.text.replace("\n", " ")[:300]
        print(f"--- {rec.source}\n{preview}")

    if args.inspect:
        return
    if not records:
        raise SystemExit("No records extracted")
    meta = write_tokens(records, args.out_dir, args.val_fraction, args.seed, args.source_url)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
