from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from reflexlm.experiment import create_experiment_run
from reflexlm.llm.qlora import QLoRAConfig, train_qlora_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Train route-specialized QLoRA adapters from by-route SFT splits.")
    parser.add_argument("--base-model-name", required=True)
    parser.add_argument("--dataset-root", required=True, help="Path to phase2_sft/<style>/by_route")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--route", action="append", help="Optional route name; repeat to restrict routes")
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
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Skip routes whose adapter output already exists; useful after an interrupted route run.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    route_dirs = (
        [dataset_root / route for route in args.route]
        if args.route
        else sorted(path for path in dataset_root.iterdir() if path.is_dir())
    )
    run = create_experiment_run(
        kind="route_expert_training",
        name="qwen_route_experts",
        config={
            "base_model_name": args.base_model_name,
            "dataset_root": str(dataset_root.resolve()),
            "routes": [path.name for path in route_dirs],
            "output_root": str(Path(args.output_root).resolve()),
        },
        run_root=args.run_root,
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    adapter_map: dict[str, str] = {}
    route_summaries: dict[str, object] = {}
    for route_dir in route_dirs:
        route_name = route_dir.name
        route_output_dir = output_root / route_name
        if args.resume_existing and (route_output_dir / "adapter_model.safetensors").exists():
            adapter_map[route_name] = str(route_output_dir.resolve())
            route_summaries[route_name] = {
                "adapter_output_dir": str(route_output_dir.resolve()),
                "skipped_existing": True,
                "reason": "existing_adapter_found",
            }
            continue
        summary = train_qlora_adapter(
            train_jsonl=route_dir / "train.jsonl",
            val_jsonl=route_dir / "val.jsonl",
            output_dir=route_output_dir,
            config=QLoRAConfig(
                base_model_name=args.base_model_name,
                adapter_name=route_name,
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
        adapter_map[route_name] = str(route_output_dir.resolve())
        route_summaries[route_name] = summary
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    adapter_map_path = output_root / "adapter_map.json"
    adapter_map_path.write_text(json.dumps(adapter_map, indent=2), encoding="utf-8")
    payload = {
        "base_model_name": args.base_model_name,
        "dataset_root": str(dataset_root.resolve()),
        "output_root": str(output_root.resolve()),
        "adapter_map_path": str(adapter_map_path.resolve()),
        "adapter_map": adapter_map,
        "routes": route_summaries,
    }
    payload["run_manifest"] = run.finalize(payload)
    run.write_json("route_expert_summary.json", payload)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
