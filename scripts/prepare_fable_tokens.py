#!/usr/bin/env python3
"""Prepare GPT-2 token streams from Glint-Research/Fable-5-traces.

This helper deliberately supports two modes:

1. --inspect: no third-party packages required. It samples the Hugging Face repo
   via HTTPS and prints schema/text examples.
2. tokenization: requires `tiktoken` and writes train.bin/val.bin compatible with
   the upstream GPT-2-pretraining notebook style.

The dataset is repo-style trace data, not a single clean table. We therefore
walk JSONL/TXT files and extract useful strings from likely message/content
fields while avoiding binary/cache noise.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.parse
import urllib.request
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DATASET_ID = "Glint-Research/Fable-5-traces"
API_URL = f"https://huggingface.co/api/datasets/{DATASET_ID}"
RAW_BASE = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/"

TEXT_KEYS = {
    "display",
    "content",
    "text",
    "message",
    "prompt",
    "completion",
    "response",
    "summary",
    "leafText",
    "reasoning",
    "result",
    "stdout",
    "stderr",
}

REDACTION_PATTERNS = [
    # Email addresses and common API/token/key shapes. This is conservative and
    # intentionally happens before classroom previews and token writing.
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<EMAIL>"),
    (re.compile(r"(?i)\b(api[_-]?key|secret|token|password|passwd|authorization)\b\s*[:=]\s*[^\s,'\"]+"), r"\1=<REDACTED>"),
    (re.compile(r"\b(sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9_]{16,}|github_pat_[A-Za-z0-9_]{16,}|xox[baprs]-[A-Za-z0-9-]{16,})\b"), "<TOKEN>"),
    (re.compile(r"/(Users|home)/[^\s:,'\"]+"), "/<LOCAL_PATH>"),
]


def redact_text(text: str) -> str:
    for pattern, repl in REDACTION_PATTERNS:
        text = pattern.sub(repl, text)
    return text


SKIP_KEYS = {
    "uuid",
    "parentUuid",
    "leafUuid",
    "sessionId",
    "timestamp",
    "cwd",
    "version",
    "type",
    "userType",
}


@dataclass
class ExtractedRecord:
    source: str
    text: str


def http_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def http_text(path: str, byte_limit: int | None = None) -> str:
    url = RAW_BASE + urllib.parse.quote(path)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read() if byte_limit is None else response.read(byte_limit)
    return data.decode("utf-8", "replace")


def list_candidate_files(max_files: int | None = None) -> list[str]:
    meta = http_json(API_URL)
    files: list[str] = []
    for sibling in meta.get("siblings", []):
        name = sibling.get("rfilename", "")
        low = name.lower()
        if not (low.endswith(".jsonl") or low.endswith(".txt") or low.endswith(".md")):
            continue
        if name == "README.md" or name.endswith("changelog.md"):
            continue
        if any(part.startswith(".") for part in name.split("/")):
            continue
        files.append(name)
    files.sort()
    if max_files is not None:
        files = files[:max_files]
    return files


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return redact_text(text.strip())


def is_useful_text(text: str) -> bool:
    text = clean_text(text)
    if len(text) < 20:
        return False
    if re.fullmatch(r"[0-9a-fA-F\-]{16,}", text):
        return False
    return True


def walk_strings(obj: Any, parent_key: str = "") -> Iterable[str]:
    if isinstance(obj, str):
        if parent_key in TEXT_KEYS or (parent_key not in SKIP_KEYS and len(obj) > 80):
            yield obj
    elif isinstance(obj, list):
        for item in obj:
            yield from walk_strings(item, parent_key)
    elif isinstance(obj, dict):
        # Prefer semantically meaningful keys first.
        for key in sorted(obj.keys(), key=lambda k: (k not in TEXT_KEYS, k)):
            if key in SKIP_KEYS:
                continue
            yield from walk_strings(obj[key], key)


def extract_from_json_line(line: str, source: str) -> list[ExtractedRecord]:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return []
    chunks = [clean_text(x) for x in walk_strings(obj)]
    chunks = [x for x in chunks if is_useful_text(x)]
    if not chunks:
        return []
    # Keep one training example per JSONL event to preserve trace-ish boundaries.
    joined = "\n".join(dict.fromkeys(chunks))
    return [ExtractedRecord(source=source, text=joined)] if is_useful_text(joined) else []


def extract_records_from_file(path: str, max_records: int | None = None) -> list[ExtractedRecord]:
    text = http_text(path)
    records: list[ExtractedRecord] = []
    low = path.lower()
    if low.endswith(".jsonl"):
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            records.extend(extract_from_json_line(line, f"{path}:{line_no}"))
            if max_records is not None and len(records) >= max_records:
                return records[:max_records]
    else:
        cleaned = clean_text(text)
        if is_useful_text(cleaned):
            records.append(ExtractedRecord(source=path, text=cleaned))
    return records[:max_records] if max_records is not None else records


def collect_records(max_files: int | None, max_records: int | None) -> list[ExtractedRecord]:
    records: list[ExtractedRecord] = []
    for path in list_candidate_files(max_files=max_files):
        remaining = None if max_records is None else max_records - len(records)
        if remaining is not None and remaining <= 0:
            break
        try:
            records.extend(extract_records_from_file(path, max_records=remaining))
        except Exception as exc:  # keep data prep robust for classroom use
            print(f"warning: failed to read {path}: {exc}", file=sys.stderr)
        if max_records is not None and len(records) >= max_records:
            break
    return records


def write_tokens(records: list[ExtractedRecord], out_dir: Path, val_fraction: float, seed: int) -> dict[str, Any]:
    try:
        import tiktoken  # type: ignore
    except Exception as exc:
        raise SystemExit("Tokenization requires `pip install tiktoken`") from exc

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
        "dataset": DATASET_ID,
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
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--out-dir", type=Path, default=Path("data/fable_tokens"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    if args.dataset != DATASET_ID:
        raise SystemExit(f"Only {DATASET_ID!r} is currently supported")

    files = list_candidate_files(max_files=args.max_files)
    print(f"dataset={DATASET_ID}")
    print(f"candidate_files={len(files)}")
    for name in files[:10]:
        print(f"file: {name}")

    records = collect_records(max_files=args.max_files, max_records=args.max_records)
    print(f"records={len(records)}")
    for rec in records[:5]:
        preview = rec.text.replace("\n", " ")[:300]
        print(f"--- {rec.source}\n{preview}")

    if args.inspect:
        return
    if not records:
        raise SystemExit("No records extracted")
    meta = write_tokens(records, args.out_dir, args.val_fraction, args.seed)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
