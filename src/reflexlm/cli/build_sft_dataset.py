from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.llm.prompts import PHASE2_PROMPT_STYLES
from reflexlm.llm.sft import materialize_sft_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 2 SFT datasets from Phase 1 JSONL splits.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prompt-style",
        action="append",
        choices=list(PHASE2_PROMPT_STYLES),
        required=True,
    )
    parser.add_argument("--synapse-checkpoint")
    parser.add_argument("--synapse-device", default="cpu")
    parser.add_argument("--run-root")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    manifest = materialize_sft_corpus(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        prompt_styles=args.prompt_style,
        synapse_checkpoint=args.synapse_checkpoint,
        synapse_device=args.synapse_device,
        run_root=args.run_root,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
