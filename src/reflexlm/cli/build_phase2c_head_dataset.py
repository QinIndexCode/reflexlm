from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.llm.head_dataset import materialize_phase2c_head_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase 2C native-head supervision datasets from Phase 1 JSONL splits."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--test-jsonl")
    parser.add_argument("--extra-train-jsonl", action="append", default=[])
    parser.add_argument("--extra-val-jsonl", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--synapse-checkpoint")
    parser.add_argument("--synapse-device", default="cpu")
    parser.add_argument("--run-root")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    manifest = materialize_phase2c_head_corpus(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        test_jsonl=args.test_jsonl,
        extra_train_jsonls=args.extra_train_jsonl,
        extra_val_jsonls=args.extra_val_jsonl,
        output_dir=args.output_dir,
        synapse_checkpoint=args.synapse_checkpoint,
        synapse_device=args.synapse_device,
        run_root=args.run_root,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
