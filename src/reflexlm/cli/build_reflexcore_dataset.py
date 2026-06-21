from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.dataset import (
    build_reflexcore_examples,
    split_examples_by_episode,
    split_hashes,
    write_reflexcore_jsonl,
)
from reflexlm.core.schema import dataset_hash
from reflexlm.data.jsonl import read_jsonl
from reflexlm.models.features import StateVectorizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a ReflexCore V0 sensory-motor JSONL dataset.",
    )
    parser.add_argument("--input-jsonl", required=True, help="Existing TrajectoryRecord JSONL")
    parser.add_argument("--output-jsonl", required=True, help="Output ReflexCore JSONL")
    parser.add_argument("--split-dir", help="Optional directory for train/val/test splits")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--hash-bins", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument("--manifest-json", help="Optional manifest path")
    args = parser.parse_args()

    vectorizer = StateVectorizer(hash_bins=args.hash_bins)
    records = read_jsonl(Path(args.input_jsonl))
    examples = build_reflexcore_examples(
        records,
        vectorizer=vectorizer,
        vocab_size=args.vocab_size,
        max_text_tokens=args.max_text_tokens,
    )
    output_path = Path(args.output_jsonl)
    write_reflexcore_jsonl(output_path, examples)
    manifest = {
        "input_jsonl": str(Path(args.input_jsonl)),
        "output_jsonl": str(output_path),
        "record_count": len(records),
        "example_count": len(examples),
        "dataset_hash": dataset_hash(examples),
        "seed": args.seed,
        "split_hashes": {},
        "vectorizer": {
            "hash_bins": args.hash_bins,
            "vector_dim": vectorizer.vector_dim,
        },
        "vocab_size": args.vocab_size,
        "max_text_tokens": args.max_text_tokens,
    }
    if args.split_dir:
        split_dir = Path(args.split_dir)
        splits = split_examples_by_episode(
            examples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        split_dir.mkdir(parents=True, exist_ok=True)
        for name, split_examples in splits.items():
            write_reflexcore_jsonl(split_dir / f"{name}.jsonl", split_examples)
        manifest["split_dir"] = str(split_dir)
        manifest["split_hashes"] = split_hashes(splits)
        manifest["split_counts"] = {name: len(items) for name, items in splits.items()}
    manifest_path = Path(args.manifest_json) if args.manifest_json else output_path.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
