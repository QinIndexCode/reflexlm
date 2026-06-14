from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.experiment import create_experiment_run
from reflexlm.train import (
    TrainerConfig,
    save_model_checkpoint,
    train_flat_text_baseline,
    train_nsi_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Phase 1 NSI model or flat baseline.")
    parser.add_argument("--dataset", required=True, help="Path to a JSONL training split")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--variant",
        choices=[
            "v1",
            "weighted",
            "route",
            "ablate_aux",
            "fast",
            "slot_focus",
            "tiny_slot",
            "micro_slot",
            "reflex_tiny",
            "reflex_micro",
        ],
        default="v1",
        help="Small-model experiment preset.",
    )
    parser.add_argument("--action-class-weighting", choices=["none", "inverse", "inverse_sqrt"])
    parser.add_argument("--hard-task-sampling-multiplier", type=float)
    parser.add_argument("--aux-loss-scale", type=float)
    parser.add_argument("--legal-action-mask", action="store_true")
    parser.add_argument("--route-conditioned-action", action="store_true")
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--encoder-depth", type=int)
    parser.add_argument("--gru-layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--command-slot-loss-weight", type=float)
    parser.add_argument("--file-slot-loss-weight", type=float)
    parser.add_argument("--disable-auxiliary-heads", action="store_true")
    parser.add_argument("--disable-action-conditioned-world-model", action="store_true")
    parser.add_argument("--residual-world-model", action="store_true")
    parser.add_argument("--hash-bins", type=int)
    parser.add_argument(
        "--disable-route-features",
        action="store_true",
        help="Zero route indicator features for ablation; does not change labels or task rules.",
    )
    parser.add_argument(
        "--disable-task-features",
        action="store_true",
        help="Zero task-family indicator features for ablation; does not change labels or task rules.",
    )
    parser.add_argument(
        "--disable-failure-signal-features",
        action="store_true",
        help="Zero visible failure summary features and their slot-match interactions.",
    )
    parser.add_argument(
        "--disable-slot-semantic-features",
        action="store_true",
        help="Zero command/file slot semantic features while keeping raw slot text hashes.",
    )
    parser.add_argument(
        "--baseline",
        choices=["nsi", "flat_text"],
        default="nsi",
    )
    parser.add_argument("--run-name", help="Optional human-readable run name")
    parser.add_argument("--run-root", help="Optional run root directory")
    parser.add_argument("--output-json", help="Optional path to write the JSON summary")
    args = parser.parse_args()
    preset = {
        "v1": {
            "action_class_weighting": "none",
            "hard_task_sampling_multiplier": 1.0,
            "aux_loss_scale": 1.0,
            "use_legal_action_mask": False,
            "route_conditioned_action": False,
        },
        "weighted": {
            "action_class_weighting": "inverse_sqrt",
            "hard_task_sampling_multiplier": 3.0,
            "aux_loss_scale": 1.0,
            "use_legal_action_mask": True,
            "route_conditioned_action": False,
        },
        "route": {
            "action_class_weighting": "inverse_sqrt",
            "hard_task_sampling_multiplier": 3.0,
            "aux_loss_scale": 1.0,
            "use_legal_action_mask": True,
            "route_conditioned_action": args.baseline == "nsi",
        },
        "ablate_aux": {
            "action_class_weighting": "inverse_sqrt",
            "hard_task_sampling_multiplier": 3.0,
            "aux_loss_scale": 0.0,
            "use_legal_action_mask": True,
            "route_conditioned_action": False,
        },
        "fast": {
            "action_class_weighting": "none",
            "hard_task_sampling_multiplier": 1.0,
            "aux_loss_scale": 0.0,
            "use_legal_action_mask": True,
            "route_conditioned_action": False,
            "hidden_dim": 64,
            "encoder_depth": 1,
            "gru_layers": 1,
            "dropout": 0.0,
            "auxiliary_heads": False,
        },
        "slot_focus": {
            "action_class_weighting": "none",
            "hard_task_sampling_multiplier": 1.0,
            "aux_loss_scale": 0.0,
            "use_legal_action_mask": True,
            "route_conditioned_action": False,
            "hidden_dim": 64,
            "encoder_depth": 1,
            "gru_layers": 1,
            "dropout": 0.0,
            "command_slot_loss_weight": 1.0,
            "file_slot_loss_weight": 0.5,
            "auxiliary_heads": False,
        },
        "tiny_slot": {
            "action_class_weighting": "none",
            "hard_task_sampling_multiplier": 1.0,
            "aux_loss_scale": 0.0,
            "use_legal_action_mask": True,
            "route_conditioned_action": False,
            "hidden_dim": 32,
            "encoder_depth": 0,
            "gru_layers": 1,
            "dropout": 0.0,
            "command_slot_loss_weight": 1.0,
            "file_slot_loss_weight": 0.5,
            "auxiliary_heads": False,
            "hash_bins": 64,
        },
        "micro_slot": {
            "action_class_weighting": "none",
            "hard_task_sampling_multiplier": 1.0,
            "aux_loss_scale": 0.0,
            "use_legal_action_mask": True,
            "route_conditioned_action": False,
            "hidden_dim": 16,
            "encoder_depth": 0,
            "gru_layers": 1,
            "dropout": 0.0,
            "command_slot_loss_weight": 1.0,
            "file_slot_loss_weight": 0.5,
            "auxiliary_heads": False,
            "hash_bins": 32,
        },
        "reflex_tiny": {
            "action_class_weighting": "none",
            "hard_task_sampling_multiplier": 1.0,
            "aux_loss_scale": 0.0,
            "use_legal_action_mask": True,
            "route_conditioned_action": False,
            "hidden_dim": 32,
            "encoder_depth": 0,
            "gru_layers": 0,
            "dropout": 0.0,
            "command_slot_loss_weight": 1.0,
            "file_slot_loss_weight": 0.5,
            "auxiliary_heads": False,
            "hash_bins": 16,
        },
        "reflex_micro": {
            "action_class_weighting": "none",
            "hard_task_sampling_multiplier": 1.0,
            "aux_loss_scale": 0.0,
            "use_legal_action_mask": True,
            "route_conditioned_action": False,
            "hidden_dim": 16,
            "encoder_depth": 0,
            "gru_layers": 0,
            "dropout": 0.0,
            "command_slot_loss_weight": 1.0,
            "file_slot_loss_weight": 0.5,
            "auxiliary_heads": False,
            "hash_bins": 0,
        },
    }[args.variant]
    action_class_weighting = args.action_class_weighting or preset["action_class_weighting"]
    hard_task_sampling_multiplier = (
        args.hard_task_sampling_multiplier
        if args.hard_task_sampling_multiplier is not None
        else preset["hard_task_sampling_multiplier"]
    )
    aux_loss_scale = args.aux_loss_scale if args.aux_loss_scale is not None else preset["aux_loss_scale"]
    use_legal_action_mask = args.legal_action_mask or bool(preset["use_legal_action_mask"])
    route_conditioned_action = args.route_conditioned_action or bool(
        preset["route_conditioned_action"]
    )
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else preset.get("hidden_dim")
    encoder_depth = (
        args.encoder_depth if args.encoder_depth is not None else preset.get("encoder_depth")
    )
    gru_layers = args.gru_layers if args.gru_layers is not None else preset.get("gru_layers")
    dropout = args.dropout if args.dropout is not None else preset.get("dropout")
    command_slot_loss_weight = (
        args.command_slot_loss_weight
        if args.command_slot_loss_weight is not None
        else preset.get("command_slot_loss_weight", 0.2)
    )
    file_slot_loss_weight = (
        args.file_slot_loss_weight
        if args.file_slot_loss_weight is not None
        else preset.get("file_slot_loss_weight", 0.2)
    )
    auxiliary_heads = (not args.disable_auxiliary_heads) and bool(
        preset.get("auxiliary_heads", True)
    )
    hash_bins = args.hash_bins if args.hash_bins is not None else preset.get("hash_bins", 256)
    trainer_config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        action_class_weighting=action_class_weighting,
        hard_task_sampling_multiplier=hard_task_sampling_multiplier,
        aux_loss_scale=aux_loss_scale,
        use_legal_action_mask=use_legal_action_mask,
        route_conditioned_action=route_conditioned_action,
        hidden_dim=hidden_dim,
        encoder_depth=encoder_depth,
        gru_layers=gru_layers,
        dropout=dropout,
        command_slot_loss_weight=command_slot_loss_weight,
        file_slot_loss_weight=file_slot_loss_weight,
        auxiliary_heads=auxiliary_heads,
        hash_bins=hash_bins,
        include_route_features=not args.disable_route_features,
        include_task_features=not args.disable_task_features,
        include_failure_signal_features=not args.disable_failure_signal_features,
        include_slot_semantic_features=not args.disable_slot_semantic_features,
        action_conditioned_world_model=not args.disable_action_conditioned_world_model,
        residual_world_model=args.residual_world_model,
    )
    run = create_experiment_run(
        kind="training",
        name=args.run_name or f"{args.baseline}_{'smoke' if args.smoke else 'full'}",
        config={
            "baseline": args.baseline,
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "seed": args.seed,
            "smoke": args.smoke,
            "variant": args.variant,
            "action_class_weighting": action_class_weighting,
            "hard_task_sampling_multiplier": hard_task_sampling_multiplier,
            "aux_loss_scale": aux_loss_scale,
            "legal_action_mask": use_legal_action_mask,
            "route_conditioned_action": route_conditioned_action,
            "hidden_dim": hidden_dim,
            "encoder_depth": encoder_depth,
            "gru_layers": gru_layers,
            "dropout": dropout,
            "command_slot_loss_weight": command_slot_loss_weight,
            "file_slot_loss_weight": file_slot_loss_weight,
            "auxiliary_heads": auxiliary_heads,
            "hash_bins": hash_bins,
            "include_route_features": not args.disable_route_features,
            "include_task_features": not args.disable_task_features,
            "include_failure_signal_features": not args.disable_failure_signal_features,
            "include_slot_semantic_features": not args.disable_slot_semantic_features,
            "action_conditioned_world_model": not args.disable_action_conditioned_world_model,
            "residual_world_model": args.residual_world_model,
        },
        run_root=args.run_root,
    )
    if args.baseline == "flat_text":
        model, vectorizer, summary = train_flat_text_baseline(
            args.dataset,
            trainer_config=trainer_config,
            smoke=args.smoke,
        )
    else:
        model, vectorizer, summary = train_nsi_model(
            args.dataset,
            trainer_config=trainer_config,
            smoke=args.smoke,
        )
    checkpoint_path = save_model_checkpoint(
        model,
        vectorizer,
        checkpoint_path=run.path / "model.pt",
        model_kind=summary["model_kind"],
        summary=summary,
    )
    summary["checkpoint_path"] = str(checkpoint_path)
    summary["run_path"] = str(run.path)
    summary["variant"] = args.variant
    summary["run_manifest"] = run.finalize(
        {
            "checkpoint_path": str(checkpoint_path),
            "dataset": args.dataset,
            "baseline": args.baseline,
        }
    )
    run.write_json("summary.json", summary)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
