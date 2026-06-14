from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.llm.qlora import QLoRAConfig, train_qlora_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 7B QLoRA adapter on the prepared SFT corpus.")
    parser.add_argument("--base-model-name", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-name", required=True)
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="4bit")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-root")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    summary = train_qlora_adapter(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        config=QLoRAConfig(
            base_model_name=args.base_model_name,
            adapter_name=args.adapter_name,
            quantization=args.quantization,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=args.max_length,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            seed=args.seed,
            device=args.device,
        ),
        run_root=args.run_root,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
