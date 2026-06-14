from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from reflexlm.cli.diagnose_phase2i_command_slots import _load_native_head_model
from reflexlm.llm.native_head_training import (
    Phase2CHeadJsonlDataset,
    _collate_head_rows,
    _finalize_action_accuracy_by_target,
    _move_batch_to_device,
    _update_action_accuracy_by_target,
)
from reflexlm.models.features import ACTION_ORDER


DEFAULT_SUMMARY = Path(
    "artifacts/reports/phase2j_source_overlap_hard_actiongate/"
    "phase2j_source_overlap_hard_actiongate_smoke.training_summary.json"
)
DEFAULT_VAL_JSONL = Path(
    "artifacts/datasets/phase2j_source_overlap_hard_actiongate_head/val.jsonl"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/phase2j_source_overlap_hard_actiongate/"
    "phase2j_source_overlap_hard_actiongate_action_confusion_diagnostic.json"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _action_name(index: int) -> str:
    if 0 <= index < len(ACTION_ORDER):
        return ACTION_ORDER[index].value
    return f"unknown:{index}"


def diagnose_native_head_actions(
    *,
    training_summary_json: str | Path,
    val_jsonl: str | Path,
    output_json: str | Path,
    adapter_dir: str | Path | None = None,
    base_model_name: str | None = None,
    quantization: str | None = None,
    device: str | None = None,
    max_length: int | None = None,
    max_records: int | None = None,
    batch_size: int = 1,
    max_misclassified_examples: int = 16,
) -> dict[str, Any]:
    summary = _load_json(training_summary_json)
    config = summary.get("config") or {}
    adapter_path = Path(adapter_dir or summary["adapter_output_dir"])
    resolved_base_model = base_model_name or summary.get("base_model_name") or config.get(
        "base_model_name"
    )
    if not resolved_base_model:
        raise ValueError("base_model_name is required when training summary does not provide it")
    resolved_quantization = quantization or config.get("quantization", "4bit")
    resolved_device = device or config.get("device", "cuda")
    resolved_max_length = int(max_length or config.get("max_length") or 512)
    resolved_max_records = max_records
    if resolved_max_records is None and config.get("max_val_records") is not None:
        resolved_max_records = int(config["max_val_records"])

    tokenizer, model, model_device = _load_native_head_model(
        adapter_dir=adapter_path,
        base_model_name=str(resolved_base_model),
        quantization=str(resolved_quantization),
        device=str(resolved_device),
    )
    head_config = model.heads.config
    dataset = Phase2CHeadJsonlDataset(val_jsonl, limit=resolved_max_records)

    def collate_with_rows(
        rows: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
        return rows, _collate_head_rows(
            tokenizer,
            rows,
            max_length=resolved_max_length,
            use_pairwise_command_reranker=bool(head_config.use_pairwise_command_reranker),
            pairwise_command_policy=head_config.pairwise_command_policy,
            pairwise_command_max_length=head_config.pairwise_command_max_length,
            pairwise_command_top_k=head_config.pairwise_command_top_k,
            command_candidate_encoder=head_config.command_candidate_encoder,
            command_candidate_feature_dim=int(head_config.command_candidate_feature_dim),
        )

    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        collate_fn=collate_with_rows,
    )
    action_accuracy_by_target: dict[str, dict[str, Any]] = {}
    misclassified_examples: list[dict[str, Any]] = []
    total = 0
    correct = 0

    with torch.inference_mode():
        for rows, batch in loader:
            batch = _move_batch_to_device(batch, model_device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                nsi_latent=batch["nsi_latent"],
                command_input_ids=batch.get("command_input_ids"),
                command_attention_mask=batch.get("command_attention_mask"),
                command_candidate_mask=batch.get("command_candidate_mask"),
                command_candidate_features=batch.get("command_candidate_features"),
                command_candidate_intents=batch.get("command_candidate_intents"),
                command_pair_input_ids=batch.get("command_pair_input_ids"),
                command_pair_attention_mask=batch.get("command_pair_attention_mask"),
                command_pair_mask=batch.get("command_pair_mask"),
                file_input_ids=batch.get("file_input_ids"),
                file_attention_mask=batch.get("file_attention_mask"),
                file_candidate_mask=batch.get("file_candidate_mask"),
            )
            predictions = outputs["action_logits"].argmax(dim=-1)
            labels = batch["action_labels"]
            _update_action_accuracy_by_target(
                action_accuracy_by_target,
                predictions=predictions,
                labels=labels,
            )
            logits = outputs["action_logits"].detach().float().cpu().tolist()
            for row, predicted_index, target_index, row_logits in zip(
                rows,
                predictions.detach().cpu().tolist(),
                labels.detach().cpu().tolist(),
                logits,
            ):
                predicted_name = _action_name(int(predicted_index))
                target_name = _action_name(int(target_index))
                is_correct = predicted_name == target_name
                total += 1
                correct += int(is_correct)
                if is_correct or len(misclassified_examples) >= max_misclassified_examples:
                    continue
                misclassified_examples.append(
                    {
                        "example_id": row.get("example_id"),
                        "episode_id": row.get("episode_id"),
                        "gold_action": target_name,
                        "pred_action": predicted_name,
                        "action_logits": {
                            _action_name(index): round(float(value), 6)
                            for index, value in enumerate(row_logits)
                        },
                        "head_scope": row.get("head_scope"),
                        "command_slot": row.get("command_slot"),
                        "file_slot": row.get("file_slot"),
                    }
                )

    report = {
        "report_family": "native_head_action_confusion_diagnostic",
        "sealed_usage": {"sealed_splits_used": False},
        "training_summary_json": str(Path(training_summary_json)),
        "val_jsonl": str(Path(val_jsonl)),
        "adapter_dir": str(adapter_path),
        "overall": {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
        },
        "action_accuracy_by_target": _finalize_action_accuracy_by_target(
            action_accuracy_by_target
        ),
        "misclassified_examples": misclassified_examples,
    }
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose action-head confusion for a saved native-head adapter."
    )
    parser.add_argument("--training-summary-json", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--val-jsonl", default=str(DEFAULT_VAL_JSONL))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--adapter-dir")
    parser.add_argument("--base-model-name")
    parser.add_argument("--quantization")
    parser.add_argument("--device")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-misclassified-examples", type=int, default=16)
    args = parser.parse_args()
    report = diagnose_native_head_actions(
        training_summary_json=args.training_summary_json,
        val_jsonl=args.val_jsonl,
        output_json=args.output_json,
        adapter_dir=args.adapter_dir,
        base_model_name=args.base_model_name,
        quantization=args.quantization,
        device=args.device,
        max_length=args.max_length,
        max_records=args.max_records,
        batch_size=args.batch_size,
        max_misclassified_examples=args.max_misclassified_examples,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
