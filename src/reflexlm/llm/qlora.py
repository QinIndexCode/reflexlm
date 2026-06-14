from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from datetime import UTC, datetime

import torch
from torch.utils.data import DataLoader, Dataset

from reflexlm.experiment import create_experiment_run


@dataclass(slots=True)
class QLoRAConfig:
    base_model_name: str
    adapter_name: str
    quantization: str = "4bit"
    learning_rate: float = 2e-4
    epochs: int = 1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 384
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 13
    device: str = "cuda"
    progress_log_interval_steps: int = 50
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


class SFTJsonlDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: str | Path) -> None:
        self.rows = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _tokenize_example(tokenizer: Any, row: dict[str, Any], *, max_length: int) -> dict[str, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_prompt"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(row["target_text"], add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    target_budget = min(len(target_ids) + 1, max_length)
    max_prompt_len = max(max_length - target_budget, 1)
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]
    target_ids = target_ids[: max_length - len(prompt_ids) - 1]
    input_ids = prompt_ids + target_ids + [eos_id]
    labels = [-100] * len(prompt_ids) + target_ids + [eos_id]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _collate(tokenizer: Any, rows: list[dict[str, Any]], *, max_length: int) -> dict[str, torch.Tensor]:
    batch = [_tokenize_example(tokenizer, row, max_length=max_length) for row in rows]
    pad_id = tokenizer.pad_token_id
    max_len = max(item["input_ids"].shape[0] for item in batch)
    collated: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in batch:
        pad = max_len - item["input_ids"].shape[0]
        collated["input_ids"].append(torch.nn.functional.pad(item["input_ids"], (0, pad), value=pad_id))
        collated["attention_mask"].append(
            torch.nn.functional.pad(item["attention_mask"], (0, pad), value=0)
        )
        collated["labels"].append(torch.nn.functional.pad(item["labels"], (0, pad), value=-100))
    return {key: torch.stack(values) for key, values in collated.items()}


def _build_quantization_config(config: QLoRAConfig) -> Any:
    from transformers import BitsAndBytesConfig

    if config.quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    if config.quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if config.quantization == "none":
        return None
    raise ValueError(f"Unsupported quantization mode: {config.quantization}")


def _write_progress_snapshot(
    run: Any,
    *,
    stage: str,
    epoch: int,
    step: int,
    total_steps: int,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "stage": stage,
        "epoch": epoch,
        "step": step,
        "total_steps": total_steps,
        "metrics": metrics,
    }
    run.write_json("progress/latest.json", payload)
    existing_rows: list[dict[str, Any]] = []
    progress_history_path = run.path / "progress" / "history.jsonl"
    if progress_history_path.exists():
        existing_rows = [
            json.loads(line)
            for line in progress_history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    existing_rows.append(payload)
    run.write_jsonl("progress/history.jsonl", existing_rows)


def train_qlora_adapter(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    output_dir: str | Path,
    config: QLoRAConfig,
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(config.seed)
    output_dir = Path(output_dir)
    run = create_experiment_run(
        kind="qlora_training",
        name=config.adapter_name,
        config={
            "train_jsonl": str(Path(train_jsonl).resolve()),
            "val_jsonl": str(Path(val_jsonl).resolve()),
            "output_dir": str(output_dir.resolve()),
            "config": asdict(config),
        },
        run_root=run_root,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        device_map="auto" if config.device == "cuda" else None,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        quantization_config=_build_quantization_config(config),
    )
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.train()
    train_dataset = SFTJsonlDataset(train_jsonl)
    val_dataset = SFTJsonlDataset(val_jsonl)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        collate_fn=lambda rows: _collate(tokenizer, rows, max_length=config.max_length),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        collate_fn=lambda rows: _collate(tokenizer, rows, max_length=config.max_length),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        print(
            json.dumps(
                {
                    "event": "epoch_start",
                    "epoch": epoch + 1,
                    "epochs": config.epochs,
                    "train_batches": len(train_loader),
                    "val_batches": len(val_loader),
                    "adapter_name": config.adapter_name,
                }
            ),
            flush=True,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss_total = 0.0
        train_batches = 0
        first_train_loss: float | None = None
        for step, batch in enumerate(train_loader, start=1):
            device = next(model.parameters()).device
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            if step % config.gradient_accumulation_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if first_train_loss is None:
                first_train_loss = float(outputs.loss.item())
            train_loss_total += float(outputs.loss.item())
            train_batches += 1
            if (
                step == 1
                or step == len(train_loader)
                or step % max(config.progress_log_interval_steps, 1) == 0
            ):
                running_train_loss = train_loss_total / max(train_batches, 1)
                progress_metrics = {
                    "running_train_loss": running_train_loss,
                    "last_batch_loss": float(outputs.loss.item()),
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "micro_batch_size": config.micro_batch_size,
                }
                _write_progress_snapshot(
                    run,
                    stage="train",
                    epoch=epoch + 1,
                    step=step,
                    total_steps=len(train_loader),
                    metrics=progress_metrics,
                )
                print(
                    json.dumps(
                        {
                            "event": "train_progress",
                            "epoch": epoch + 1,
                            "step": step,
                            "total_steps": len(train_loader),
                            **progress_metrics,
                        }
                    ),
                    flush=True,
                )
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.inference_mode():
            for val_step, batch in enumerate(val_loader, start=1):
                device = next(model.parameters()).device
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                val_loss_total += float(outputs.loss.item())
                val_batches += 1
                if (
                    val_step == 1
                    or val_step == len(val_loader)
                    or val_step % max(config.progress_log_interval_steps, 1) == 0
                ):
                    progress_metrics = {
                        "running_val_loss": val_loss_total / max(val_batches, 1),
                        "last_batch_loss": float(outputs.loss.item()),
                    }
                    _write_progress_snapshot(
                        run,
                        stage="validation",
                        epoch=epoch + 1,
                        step=val_step,
                        total_steps=len(val_loader),
                        metrics=progress_metrics,
                    )
        history.append(
            {
                "epoch": epoch + 1,
                "first_train_loss": float(first_train_loss or 0.0),
                "train_loss": train_loss_total / max(train_batches, 1),
                "val_loss": val_loss_total / max(val_batches, 1),
            }
        )
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    "epoch": epoch + 1,
                    "train_loss": history[-1]["train_loss"],
                    "val_loss": history[-1]["val_loss"],
                }
            ),
            flush=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary = {
        "base_model_name": config.base_model_name,
        "adapter_name": config.adapter_name,
        "adapter_output_dir": str(output_dir.resolve()),
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "history": history,
        "config": asdict(config),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    summary["run_manifest"] = run.finalize(summary)
    run.write_json("training_summary.json", summary)
    return summary
