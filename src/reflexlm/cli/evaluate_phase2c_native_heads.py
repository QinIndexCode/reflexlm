from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.llm.native_head_training import evaluate_native_head_adapter


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Phase 2C native-head adapter on a head JSONL split without retraining."
    )
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--base-model-name")
    parser.add_argument("--training-summary-json")
    parser.add_argument("--eval-split", default="eval")
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="4bit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-eval-records", type=int)
    parser.add_argument("--include-prediction-records", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    training_summary = _read_json(args.training_summary_json)
    train_config = training_summary.get("config") if isinstance(training_summary.get("config"), dict) else {}
    base_model_name = args.base_model_name or training_summary.get("base_model_name")
    if not base_model_name:
        raise SystemExit("--base-model-name is required when --training-summary-json does not provide it")
    max_length = args.max_length or train_config.get("max_length")
    loss_weights = train_config.get("loss_weights") if isinstance(train_config.get("loss_weights"), dict) else None
    report = evaluate_native_head_adapter(
        eval_jsonl=args.eval_jsonl,
        adapter_dir=args.adapter_dir,
        base_model_name=str(base_model_name),
        quantization=args.quantization,
        device=args.device,
        max_length=int(max_length) if max_length else None,
        batch_size=args.batch_size,
        eval_split=args.eval_split,
        max_eval_records=args.max_eval_records,
        loss_weights=loss_weights,
        include_prediction_records=args.include_prediction_records,
    )
    if args.training_summary_json:
        report["training_summary_json"] = str(Path(args.training_summary_json))
        report["training_summary_config_hash"] = training_summary.get("config_hash")
        report["training_summary_effective_split_hashes"] = training_summary.get(
            "effective_split_hashes"
        )
    if args.output_json:
        _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
