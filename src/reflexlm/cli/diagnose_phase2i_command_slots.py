from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from reflexlm.llm.candidate_features import source_overlap_command_slot_prediction
from reflexlm.llm.candidate_features import (
    COMMAND_CANDIDATE_FEATURE_GROUP_RANGES,
    normalize_command_candidate_feature_ablation_groups,
)
from reflexlm.llm.native_cortex import NativeCortexHeadConfig, QwenBackboneHeadAdapter
from reflexlm.llm.native_head_training import (
    NativeHeadTrainConfig,
    Phase2CHeadJsonlDataset,
    _build_quantization_config,
    _collate_head_rows,
    _move_batch_to_device,
)


LOGIT_SOURCES = {
    "slot_head": "command_slot_logits",
    "lightweight_candidate": "command_lightweight_candidate_logits",
    "pairwise": "command_pair_logits",
    "effective": "command_candidate_logits",
}


def _zero_command_candidate_feature_groups_tensor(
    features: torch.Tensor,
    groups: list[str] | tuple[str, ...] | None,
) -> torch.Tensor:
    normalized = normalize_command_candidate_feature_ablation_groups(groups)
    if not normalized:
        return features
    output = features.clone()
    width = int(output.shape[-1])
    for group in normalized:
        for start, end in COMMAND_CANDIDATE_FEATURE_GROUP_RANGES[group]:
            output[..., start : min(end, width)] = 0.0
    return output


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def _counter_rows(counter: Counter[tuple[int, int]]) -> list[dict[str, int]]:
    return [
        {"gold_slot": gold, "pred_slot": pred, "count": count}
        for (gold, pred), count in sorted(counter.items())
    ]


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    total = int(bucket.get("total", 0))
    correct = int(bucket.get("correct", 0))
    return {
        "total": total,
        "correct": correct,
        "accuracy": _accuracy(correct, total),
        "predicted_slots": dict(sorted(bucket.get("predicted_slots", Counter()).items())),
    }


def summarize_command_slot_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize command-slot prediction records without assuming policy details."""

    source_totals: dict[str, dict[str, Any]] = {}
    confusion: dict[str, Counter[tuple[int, int]]] = defaultdict(Counter)
    by_intent: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "correct": 0, "predicted_slots": Counter()})
    )
    by_gold_slot: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "correct": 0, "predicted_slots": Counter()})
    )

    for record in records:
        gold = int(record["gold_slot"])
        intent = str(record.get("command_intent") or "unknown")
        for source, pred_value in record.get("predictions", {}).items():
            pred = int(pred_value)
            if source not in source_totals:
                source_totals[source] = {"total": 0, "correct": 0, "predicted_slots": Counter()}
            source_totals[source]["total"] += 1
            source_totals[source]["correct"] += int(pred == gold)
            source_totals[source]["predicted_slots"][pred] += 1
            confusion[source][(gold, pred)] += 1

            intent_bucket = by_intent[source][intent]
            intent_bucket["total"] += 1
            intent_bucket["correct"] += int(pred == gold)
            intent_bucket["predicted_slots"][pred] += 1

            slot_bucket = by_gold_slot[source][str(gold)]
            slot_bucket["total"] += 1
            slot_bucket["correct"] += int(pred == gold)
            slot_bucket["predicted_slots"][pred] += 1

    return {
        "sources": {
            source: {
                **_finalize_bucket(bucket),
                "confusion": _counter_rows(confusion[source]),
                "by_intent": {
                    intent: _finalize_bucket(intent_bucket)
                    for intent, intent_bucket in sorted(by_intent[source].items())
                },
                "by_gold_slot": {
                    slot: _finalize_bucket(slot_bucket)
                    for slot, slot_bucket in sorted(by_gold_slot[source].items())
                },
            }
            for source, bucket in sorted(source_totals.items())
        },
        "command_record_count": len(records),
    }


def _load_native_head_model(
    *,
    adapter_dir: Path,
    base_model_name: str,
    quantization: str,
    device: str,
) -> tuple[Any, QwenBackboneHeadAdapter, torch.device]:
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    tokenizer_path = adapter_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path) if tokenizer_path.exists() else base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loader_config = NativeHeadTrainConfig(
        base_model_name=base_model_name,
        adapter_name=adapter_dir.name,
        quantization=quantization,
        device=device,
    )
    backbone = AutoModel.from_pretrained(
        base_model_name,
        device_map="auto" if device == "cuda" else None,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        quantization_config=_build_quantization_config(loader_config),
    )
    backbone = PeftModel.from_pretrained(backbone, adapter_dir / "backbone_adapter")
    head_config = NativeCortexHeadConfig(**_load_json(adapter_dir / "head_config.json"))
    model = QwenBackboneHeadAdapter(backbone, head_config=head_config)
    try:
        head_state = torch.load(adapter_dir / "native_heads.pt", map_location="cpu", weights_only=True)
    except TypeError:
        head_state = torch.load(adapter_dir / "native_heads.pt", map_location="cpu")
    model.heads.load_state_dict(head_state, strict=False)
    resolved_device = next(model.backbone.parameters()).device
    model.heads.to(resolved_device)
    model.eval()
    return tokenizer, model, resolved_device


def _logit_values(logits: torch.Tensor, row_index: int) -> list[float]:
    return [round(float(value), 6) for value in logits.detach().float().cpu()[row_index].tolist()]


def diagnose_command_slots(
    *,
    adapter_dir: str | Path,
    val_jsonl: str | Path,
    output_json: str | Path,
    training_summary: str | Path | None = None,
    base_model_name: str | None = None,
    quantization: str | None = None,
    device: str = "cuda",
    max_length: int | None = None,
    max_records: int | None = None,
    batch_size: int = 1,
    include_records: bool = True,
    zero_nsi_latent: bool = False,
    zero_command_candidate_feature_groups: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    adapter_path = Path(adapter_dir)
    summary = _load_json(training_summary) if training_summary else {}
    config = summary.get("config", {})
    resolved_base_model = base_model_name or summary.get("base_model_name") or config.get("base_model_name")
    if not resolved_base_model:
        raise ValueError("base_model_name is required when training_summary is not provided")
    resolved_quantization = quantization or config.get("quantization", "4bit")
    resolved_max_length = int(max_length or config.get("max_length") or 512)
    resolved_max_records = max_records
    if resolved_max_records is None and config.get("max_val_records") is not None:
        resolved_max_records = int(config["max_val_records"])

    tokenizer, model, resolved_device = _load_native_head_model(
        adapter_dir=adapter_path,
        base_model_name=str(resolved_base_model),
        quantization=str(resolved_quantization),
        device=device,
    )
    use_pairwise = bool(model.heads.config.use_pairwise_command_reranker)
    pairwise_policy = model.heads.config.pairwise_command_policy
    pairwise_max_length = int(model.heads.config.pairwise_command_max_length or resolved_max_length)
    pairwise_top_k = model.heads.config.pairwise_command_top_k
    command_candidate_encoder = model.heads.config.command_candidate_encoder
    command_feature_dim = int(model.heads.config.command_candidate_feature_dim)
    dataset = Phase2CHeadJsonlDataset(val_jsonl, limit=resolved_max_records)
    feature_groups = list(zero_command_candidate_feature_groups or [])
    if zero_nsi_latent and "candidate_identity" not in feature_groups:
        # Candidate identity features are derived from NSI command-identity
        # references in offline head rows. A no-NSI diagnostic must remove
        # both the latent tensor and this lightweight NSI-derived side channel.
        feature_groups.append("candidate_identity")
    feature_groups = list(normalize_command_candidate_feature_ablation_groups(feature_groups))

    def collate_with_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
        return rows, _collate_head_rows(
            tokenizer,
            rows,
            max_length=resolved_max_length,
            use_pairwise_command_reranker=use_pairwise,
            pairwise_command_policy=pairwise_policy,
            pairwise_command_max_length=pairwise_max_length,
            pairwise_command_top_k=pairwise_top_k,
            command_candidate_encoder=command_candidate_encoder,
            command_candidate_feature_dim=command_feature_dim,
        )

    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        collate_fn=collate_with_rows,
    )
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch_rows, batch in loader:
            batch = _move_batch_to_device(batch, resolved_device)
            nsi_latent = batch["nsi_latent"]
            if zero_nsi_latent:
                nsi_latent = torch.zeros_like(nsi_latent)
            command_candidate_features = batch.get("command_candidate_features")
            if command_candidate_features is not None and feature_groups:
                command_candidate_features = _zero_command_candidate_feature_groups_tensor(
                    command_candidate_features,
                    feature_groups,
                )
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                nsi_latent=nsi_latent,
                command_input_ids=batch.get("command_input_ids"),
                command_attention_mask=batch.get("command_attention_mask"),
                command_candidate_mask=batch.get("command_candidate_mask"),
                command_candidate_features=command_candidate_features,
                command_candidate_intents=batch.get("command_candidate_intents"),
                command_pair_input_ids=batch.get("command_pair_input_ids"),
                command_pair_attention_mask=batch.get("command_pair_attention_mask"),
                command_pair_mask=batch.get("command_pair_mask"),
                file_input_ids=batch.get("file_input_ids"),
                file_attention_mask=batch.get("file_attention_mask"),
                file_candidate_mask=batch.get("file_candidate_mask"),
            )
            labels = batch["command_slot_labels"].detach().cpu()
            for row_index, row in enumerate(batch_rows):
                gold = int(labels[row_index].item())
                if gold == -100:
                    continue
                predictions: dict[str, int] = {}
                logits_by_source: dict[str, list[float]] = {}
                for source, key in LOGIT_SOURCES.items():
                    if key not in outputs:
                        continue
                    logits = outputs[key]
                    predictions[source] = int(logits.detach().cpu()[row_index].argmax().item())
                    logits_by_source[source] = _logit_values(logits, row_index)
                candidates = list(row.get("candidate_commands") or [])
                predictions["source_overlap_baseline"] = source_overlap_command_slot_prediction(
                    str(row["state_prompt"]),
                    candidates,
                )
                records.append(
                    {
                        "example_id": row.get("example_id"),
                        "episode_id": row.get("episode_id"),
                        "task_type": row.get("task_type"),
                        "head_scope": row.get("head_scope"),
                        "command_intent": row.get("command_intent"),
                        "gold_slot": gold,
                        "gold_command": row.get("command"),
                        "candidate_count": len(candidates),
                        "candidate_commands": candidates,
                        "predictions": predictions,
                        "logits": logits_by_source,
                    }
                )

    summary_payload = summarize_command_slot_records(records)
    report = {
        "analysis": "phase2i_command_slot_diagnostics",
        "adapter_dir": str(adapter_path),
        "val_jsonl": str(val_jsonl),
        "training_summary": str(training_summary) if training_summary else None,
        "sealed_data_used_for_training_or_tuning": False,
        "base_model_name": str(resolved_base_model),
        "quantization": resolved_quantization,
        "device": str(resolved_device),
        "max_length": resolved_max_length,
        "max_records": resolved_max_records,
        "batch_size": max(1, int(batch_size)),
        "use_pairwise_command_reranker": use_pairwise,
        "pairwise_command_fusion": model.heads.config.pairwise_command_fusion,
        "pairwise_command_policy": pairwise_policy,
        "pairwise_command_max_length": pairwise_max_length,
        "pairwise_command_top_k": pairwise_top_k,
        "command_candidate_encoder": command_candidate_encoder,
        "command_candidate_feature_dim": command_feature_dim,
        "zero_nsi_latent": zero_nsi_latent,
        "zero_command_candidate_feature_groups": feature_groups,
        **summary_payload,
    }
    if include_records:
        report["records"] = records

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose Phase2I command-slot failures on non-sealed native-head validation data."
    )
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--training-summary")
    parser.add_argument("--base-model-name")
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-records", action="store_true")
    parser.add_argument("--zero-nsi-latent", action="store_true")
    parser.add_argument(
        "--zero-command-candidate-feature-groups",
        default="",
        help="Comma-separated command candidate feature groups to zero during diagnostics.",
    )
    args = parser.parse_args()
    report = diagnose_command_slots(
        adapter_dir=args.adapter_dir,
        val_jsonl=args.val_jsonl,
        output_json=args.output_json,
        training_summary=args.training_summary,
        base_model_name=args.base_model_name,
        quantization=args.quantization,
        device=args.device,
        max_length=args.max_length,
        max_records=args.max_records,
        batch_size=args.batch_size,
        include_records=not args.no_records,
        zero_nsi_latent=args.zero_nsi_latent,
        zero_command_candidate_feature_groups=[
            item.strip()
            for item in str(args.zero_command_candidate_feature_groups).split(",")
            if item.strip()
        ],
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
