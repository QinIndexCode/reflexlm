from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.llm.native_head_training import NativeHeadTrainConfig, train_native_head_adapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Phase 2C native action heads with a shared Qwen backbone and NSI latent side inputs."
    )
    parser.add_argument("--base-model-name", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-name", required=True)
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="4bit")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    parser.add_argument("--head-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-val-records", type=int)
    parser.add_argument("--progress-log-interval-steps", type=int, default=50)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=0)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--command-intent-loss-weight", type=float)
    parser.add_argument("--command-slot-loss-weight", type=float)
    parser.add_argument("--patch-operation-loss-weight", type=float)
    parser.add_argument("--patch-template-slot-loss-weight", type=float)
    parser.add_argument("--patch-target-file-slot-loss-weight", type=float)
    parser.add_argument("--debug-command-oversample", type=int, default=1)
    parser.add_argument("--balance-command-slots", action="store_true")
    parser.add_argument("--balance-debug-command-intents", action="store_true")
    parser.add_argument("--balance-patch-descriptor-labels", action="store_true")
    parser.add_argument("--use-pairwise-command-reranker", action="store_true")
    parser.add_argument("--pairwise-command-fusion", choices=["replace", "residual"], default="residual")
    parser.add_argument("--pairwise-command-policy", choices=["all", "ambiguous_intent"], default="all")
    parser.add_argument("--pairwise-command-max-length", type=int)
    parser.add_argument("--pairwise-command-top-k", type=int)
    parser.add_argument("--command-identity-logit-bias", type=float, default=0.0)
    parser.add_argument("--command-candidate-encoder", choices=["backbone", "features_only"], default="backbone")
    parser.add_argument("--latent-fusion", choices=["concat", "additive"], default="additive")
    parser.add_argument("--open-repair-heads-enabled", action="store_true")
    parser.add_argument("--run-root")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    config = NativeHeadTrainConfig(
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
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        seed=args.seed,
        device=args.device,
        max_train_records=args.max_train_records,
        max_val_records=args.max_val_records,
        progress_log_interval_steps=args.progress_log_interval_steps,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        checkpoint_dir=args.checkpoint_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        debug_command_oversample=args.debug_command_oversample,
        balance_command_slots=args.balance_command_slots,
        balance_debug_command_intents=args.balance_debug_command_intents,
        balance_patch_descriptor_labels=args.balance_patch_descriptor_labels,
        use_pairwise_command_reranker=args.use_pairwise_command_reranker,
        pairwise_command_fusion=args.pairwise_command_fusion,
        pairwise_command_policy=args.pairwise_command_policy,
        pairwise_command_max_length=args.pairwise_command_max_length,
        pairwise_command_top_k=args.pairwise_command_top_k,
        command_identity_logit_bias=args.command_identity_logit_bias,
        command_candidate_encoder=args.command_candidate_encoder,
        latent_fusion=args.latent_fusion,
        open_repair_heads_enabled=args.open_repair_heads_enabled,
    )
    if args.command_intent_loss_weight is not None:
        config.loss_weights["command_intent"] = args.command_intent_loss_weight
    if args.command_slot_loss_weight is not None:
        config.loss_weights["command_slot"] = args.command_slot_loss_weight
    if args.patch_operation_loss_weight is not None:
        config.loss_weights["patch_operation"] = args.patch_operation_loss_weight
    if args.patch_template_slot_loss_weight is not None:
        config.loss_weights["patch_template_slot"] = args.patch_template_slot_loss_weight
    if args.patch_target_file_slot_loss_weight is not None:
        config.loss_weights["patch_target_file_slot"] = args.patch_target_file_slot_loss_weight

    summary = train_native_head_adapter(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        config=config,
        run_root=args.run_root,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
