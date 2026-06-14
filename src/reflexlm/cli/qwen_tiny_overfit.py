from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.llm.qlora import QLoRAConfig, _build_quantization_config, train_qlora_adapter
from reflexlm.schema import action_name_to_enum


def _read_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if len(rows) >= limit:
            break
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    try:
        action_name_to_enum(str(payload.get("action", "")).strip().upper())
    except ValueError:
        return None
    if "command" not in payload or "file_target" not in payload:
        return None
    return payload


def _section_items(prompt: str, section_name: str) -> list[str]:
    lines = prompt.splitlines()
    items: list[str] = []
    capture = False
    for line in lines:
        stripped = line.strip()
        if stripped == section_name:
            capture = True
            continue
        if capture:
            if not stripped:
                break
            if stripped.startswith("- "):
                value = stripped[2:].strip()
                if value != "<none>":
                    items.append(value)
    return items


def _payload_uses_allowlisted_slots(row: dict[str, Any], payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    action = str(payload.get("action", "")).strip().upper()
    if action == "RUN_COMMAND":
        return payload.get("command") in _section_items(row["user_prompt"], "Candidate commands:")
    if action == "READ_FILE":
        return payload.get("file_target") in _section_items(row["user_prompt"], "Candidate files:")
    return payload.get("command") is None and payload.get("file_target") is None


def _generate_probe(
    *,
    base_model_name: str,
    adapter_path: Path,
    row: dict[str, Any],
    quantization: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        quantization_config=_build_quantization_config(
            QLoRAConfig(base_model_name=base_model_name, adapter_name="probe", quantization=quantization)
        ),
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_prompt"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    parsed = _extract_json_object(text)
    return {
        "generated_text": text,
        "parsed_action": parsed,
        "json_action_valid": parsed is not None,
        "allowlisted_slots_valid": _payload_uses_allowlisted_slots(row, parsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny-overfit gate before full 0.5B LoRA.")
    parser.add_argument("--base-model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-root", default="artifacts/qwen05b_tiny_overfit")
    parser.add_argument("--run-root", default="artifacts/runs_small_model")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--min-loss-drop", type=float, default=0.20)
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="4bit")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-allowlist-valid-rate", type=float, default=0.98)
    args = parser.parse_args()

    source_path = Path(args.source_jsonl)
    rows = _read_rows(source_path, args.max_examples)
    if len(rows) < 16:
        raise ValueError("Tiny-overfit gate requires at least 16 examples.")
    split_index = max(1, int(len(rows) * 0.8))
    output_root = Path(args.output_root)
    train_path = output_root / "tiny_train.jsonl"
    val_path = output_root / "tiny_val.jsonl"
    adapter_path = output_root / "adapter"
    _write_jsonl(train_path, rows[:split_index])
    _write_jsonl(val_path, rows[split_index:] or rows[:1])

    summary = train_qlora_adapter(
        train_jsonl=train_path,
        val_jsonl=val_path,
        output_dir=adapter_path,
        config=QLoRAConfig(
            base_model_name=args.base_model_name,
            adapter_name="qwen05b_tiny_overfit",
            quantization=args.quantization,
            epochs=args.epochs,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_length=args.max_length,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
        ),
        run_root=args.run_root,
    )
    first_loss = float(summary["history"][0].get("first_train_loss", 0.0))
    final_loss = float(summary["history"][-1].get("train_loss", 0.0))
    loss_drop = (first_loss - final_loss) / first_loss if first_loss > 0 else 0.0
    probe = _generate_probe(
        base_model_name=args.base_model_name,
        adapter_path=adapter_path,
        row=rows[split_index] if split_index < len(rows) else rows[0],
        quantization=args.quantization,
        max_new_tokens=args.max_new_tokens,
    )
    allowlist_valid_rate = 1.0 if probe["allowlisted_slots_valid"] else 0.0
    payload = {
        "passed": (
            loss_drop >= args.min_loss_drop
            and probe["json_action_valid"]
            and allowlist_valid_rate >= args.min_allowlist_valid_rate
        ),
        "min_loss_drop": args.min_loss_drop,
        "min_allowlist_valid_rate": args.min_allowlist_valid_rate,
        "allowlist_valid_rate": allowlist_valid_rate,
        "first_train_loss": first_loss,
        "final_train_loss": final_loss,
        "loss_drop": loss_drop,
        "probe": probe,
        "training_summary": summary,
        "train_jsonl": str(train_path.resolve()),
        "val_jsonl": str(val_path.resolve()),
        "adapter_path": str(adapter_path.resolve()),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
