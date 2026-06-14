from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.baselines.text_policies import HuggingFaceJSONPolicy
from reflexlm.eval import (
    RuleOraclePolicyAdapter,
    SequenceModelPolicy,
    evaluate_policy,
    evaluation_summary_to_dict,
)
from reflexlm.experiment import create_experiment_run
from reflexlm.llm.hybrid import AdapterJSONPolicy, HybridPolicyConfig, HybridSynapticPolicy
from reflexlm.llm.native_head_policy import NativeHeadPolicy
from reflexlm.llm.native_nervous_package import NativeNervousPolicyPackage
from reflexlm.llm.prompts import PHASE2_PROMPT_STYLES
from reflexlm.schema import TaskType
from reflexlm.train import load_model_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Phase 1 policy scaffold.")
    parser.add_argument(
        "--policy",
        choices=[
            "rule_oracle",
            "nsi_checkpoint",
            "flat_checkpoint",
            "prompt_only",
            "react",
            "qwen_adapter",
            "hybrid_synaptic_qwen",
            "qwen_native_heads",
            "phase2d_native_package",
        ],
        default="rule_oracle",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a JSONL split used to reconstruct the fixed evaluation episodes",
    )
    parser.add_argument("--checkpoint", help="Checkpoint path for nsi_checkpoint or flat_checkpoint")
    parser.add_argument("--model-name", help="Model name or local path for prompt_only or react")
    parser.add_argument("--adapter-path", help="Optional PEFT adapter path for qwen_adapter")
    parser.add_argument("--native-head-path", help="Phase 2C native-head adapter directory")
    parser.add_argument("--policy-package-path", help="Phase 2D native nervous package directory or manifest")
    parser.add_argument("--adapter-map-json", help="Optional JSON file mapping route names to adapter paths")
    parser.add_argument("--nsi-checkpoint", help="NSI checkpoint required for hybrid_synaptic_qwen")
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--native-head-max-length", type=int, default=512)
    parser.add_argument("--max-time-s", type=float, default=20.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--parse-retry-growth",
        type=float,
        default=1.0,
        help="Multiplier applied to max_new_tokens and max_time_s on each JSON parse retry.",
    )
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--policy-label", help="Optional label written into the report")
    parser.add_argument(
        "--prompt-style",
        choices=list(PHASE2_PROMPT_STYLES),
        help="Prompt style for qwen_adapter or hybrid_synaptic_qwen policies",
    )
    parser.add_argument("--limit-episodes", type=int, help="Optional cap on evaluation episodes")
    parser.add_argument(
        "--env-profile",
        default="default",
        help="Environment reconstruction profile for closed-loop evaluation",
    )
    parser.add_argument(
        "--balanced-limit",
        action="store_true",
        help="Round-robin limited evaluation episodes across selected task families",
    )
    parser.add_argument(
        "--task-filter",
        action="append",
        default=[],
        help="Task family to evaluate. May be repeated or comma-separated.",
    )
    parser.add_argument("--device", default="cpu", help="Device for checkpoint-based small models")
    parser.add_argument("--legal-action-mask", action="store_true")
    parser.add_argument("--nsi-device", default="cpu")
    parser.add_argument("--zero-nsi-latent", action="store_true")
    parser.add_argument("--disable-continuation-cache", action="store_true")
    parser.add_argument(
        "--continuation-control",
        choices=("normal", "cache_erased", "wrong_cache"),
        default=None,
        help="Runtime continuation-memory intervention for package/native-head policies.",
    )
    parser.add_argument("--disable-native-head-calls", action="store_true")
    parser.add_argument(
        "--disable-command-candidate-feature-group",
        action="append",
        default=[],
        help="Candidate feature group to zero at runtime; may be repeated or comma-separated.",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.72)
    parser.add_argument("--prediction-error-threshold", type=float, default=0.45)
    parser.add_argument("--risk-threshold", type=float, default=0.7)
    parser.add_argument("--run-name", help="Optional human-readable run name")
    parser.add_argument("--run-root", help="Optional run root directory")
    parser.add_argument("--output-json", help="Optional path to write the JSON summary")
    args = parser.parse_args()
    task_filter_values: list[TaskType] = []
    for raw_filter in args.task_filter:
        for value in raw_filter.split(","):
            value = value.strip()
            if value:
                task_filter_values.append(TaskType(value))
    task_filter = set(task_filter_values) if task_filter_values else None
    requested_continuation_control = args.continuation_control
    continuation_control = requested_continuation_control or "normal"
    if args.disable_continuation_cache:
        if requested_continuation_control is not None:
            raise ValueError(
                "--disable-continuation-cache cannot be combined with an explicit "
                "--continuation-control"
            )
        continuation_control = "cache_erased"

    if args.policy == "rule_oracle":
        policy = RuleOraclePolicyAdapter()
    elif args.policy in {"nsi_checkpoint", "flat_checkpoint"}:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for checkpoint-based policies")
        model, vectorizer, checkpoint_payload = load_model_checkpoint(args.checkpoint, device=args.device)
        policy = SequenceModelPolicy(
            model,
            vectorizer,
            policy_label=args.policy_label
            or f"{checkpoint_payload['model_kind']}_small_model",
            use_legal_action_mask=args.legal_action_mask,
            training_summary=checkpoint_payload.get("training_summary", {}),
        )
    else:
        if args.policy in {"prompt_only", "react"} and not args.model_name:
            raise ValueError("--model-name is required for prompt_only and react policies")
        if args.policy in {"prompt_only", "react"}:
            policy = HuggingFaceJSONPolicy(
                args.model_name,
                react_style=args.policy == "react",
                quantization=args.quantization,
                max_new_tokens=args.max_new_tokens,
                max_time_s=args.max_time_s,
                max_retries=args.max_retries,
                parse_retry_growth=args.parse_retry_growth,
                cpu_offload=args.cpu_offload,
                policy_label=args.policy_label,
            )
        elif args.policy == "qwen_adapter":
            if not args.model_name:
                raise ValueError("--model-name is required for qwen_adapter")
            adapter_map = (
                json.loads(Path(args.adapter_map_json).read_text(encoding="utf-8"))
                if args.adapter_map_json
                else None
            )
            policy = AdapterJSONPolicy(
                args.model_name,
                adapter_path=args.adapter_path,
                adapter_map=adapter_map,
                prompt_style=args.prompt_style
                or ("synapse_augmented" if args.adapter_path or adapter_map else "prompt_only"),
                policy_label=args.policy_label or "qwen_adapter",
                quantization=args.quantization,
                max_new_tokens=args.max_new_tokens,
                max_time_s=args.max_time_s,
                max_retries=args.max_retries,
                parse_retry_growth=args.parse_retry_growth,
                cpu_offload=args.cpu_offload,
            )
        elif args.policy == "qwen_native_heads":
            if not args.model_name or not args.native_head_path or not args.nsi_checkpoint:
                raise ValueError(
                    "--model-name, --native-head-path, and --nsi-checkpoint are required for qwen_native_heads"
                )
            policy = NativeHeadPolicy(
                base_model_name=args.model_name,
                native_head_path=args.native_head_path,
                nsi_checkpoint_path=args.nsi_checkpoint,
                quantization=args.quantization,
                nsi_device=args.nsi_device,
                device=args.device,
                cpu_offload=args.cpu_offload,
                max_length=args.native_head_max_length,
                policy_label=args.policy_label or "qwen_native_heads",
                zero_nsi_latent=args.zero_nsi_latent,
                enable_debug_continuation=continuation_control != "cache_erased",
                continuation_control=continuation_control,
                enable_native_head_calls=not args.disable_native_head_calls,
                disabled_command_candidate_feature_groups=(
                    args.disable_command_candidate_feature_group
                ),
            )
        elif args.policy == "phase2d_native_package":
            if not args.policy_package_path:
                raise ValueError("--policy-package-path is required for phase2d_native_package")
            policy = NativeNervousPolicyPackage(
                args.policy_package_path,
                continuation_control=(
                    continuation_control
                    if args.disable_continuation_cache
                    or requested_continuation_control is not None
                    else None
                ),
                disabled_command_candidate_feature_groups=(
                    args.disable_command_candidate_feature_group
                    if args.disable_command_candidate_feature_group
                    else None
                ),
            )
        else:
            if not args.model_name or not args.nsi_checkpoint:
                raise ValueError("--model-name and --nsi-checkpoint are required for hybrid_synaptic_qwen")
            adapter_map = (
                json.loads(Path(args.adapter_map_json).read_text(encoding="utf-8"))
                if args.adapter_map_json
                else {}
            )
            policy = HybridSynapticPolicy(
                nsi_checkpoint_path=args.nsi_checkpoint,
                hybrid_config=HybridPolicyConfig(
                    base_model_name=args.model_name,
                    shared_adapter_path=args.adapter_path,
                    adapter_map=adapter_map,
                    quantization=args.quantization,
                    confidence_threshold=args.confidence_threshold,
                    prediction_error_threshold=args.prediction_error_threshold,
                    risk_threshold=args.risk_threshold,
                    prompt_style=args.prompt_style or "synapse_augmented",
                    max_new_tokens=args.max_new_tokens,
                    max_time_s=args.max_time_s,
                    max_retries=args.max_retries,
                    parse_retry_growth=args.parse_retry_growth,
                    cpu_offload=args.cpu_offload,
                ),
                nsi_device=args.nsi_device,
            )

    run = create_experiment_run(
        kind="evaluation",
        name=args.run_name or args.policy,
        config={
            "policy": args.policy,
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "model_name": args.model_name,
            "adapter_path": args.adapter_path,
            "native_head_path": args.native_head_path,
            "policy_package_path": args.policy_package_path,
            "adapter_map_json": args.adapter_map_json,
            "nsi_checkpoint": args.nsi_checkpoint,
            "quantization": args.quantization,
            "max_new_tokens": args.max_new_tokens,
            "native_head_max_length": args.native_head_max_length,
            "max_time_s": args.max_time_s,
            "max_retries": args.max_retries,
            "parse_retry_growth": args.parse_retry_growth,
            "cpu_offload": args.cpu_offload,
            "policy_label": args.policy_label,
            "prompt_style": args.prompt_style,
            "limit_episodes": args.limit_episodes,
            "env_profile": args.env_profile,
            "balanced_limit": args.balanced_limit,
            "task_filter": [task.value for task in task_filter_values],
            "device": args.device,
            "legal_action_mask": args.legal_action_mask,
            "nsi_device": args.nsi_device,
            "zero_nsi_latent": args.zero_nsi_latent,
            "continuation_cache_enabled": continuation_control != "cache_erased",
            "continuation_control": continuation_control,
            "native_head_calls_enabled": not args.disable_native_head_calls,
            "disabled_command_candidate_feature_groups": (
                args.disable_command_candidate_feature_group
            ),
            "confidence_threshold": args.confidence_threshold,
            "prediction_error_threshold": args.prediction_error_threshold,
            "risk_threshold": args.risk_threshold,
        },
        run_root=args.run_root,
    )
    summary = evaluate_policy(
        policy,
        dataset_path=args.dataset,
        limit_episodes=args.limit_episodes,
        task_filter=task_filter,
        balanced_limit=args.balanced_limit,
        env_profile=args.env_profile,
        progress_dir=run.path / "progress",
    )
    payload = evaluation_summary_to_dict(summary)
    payload["run_path"] = str(run.path)
    payload["run_manifest"] = run.finalize(
        {
            "policy": args.policy,
            "dataset": str(Path(args.dataset).resolve()),
            "policy_label": payload["policy"]["policy_label"],
        }
    )
    run.write_jsonl("episode_results.jsonl", summary.per_episode)
    run.write_jsonl("trace_rows.jsonl", summary.trace_rows)
    run.write_json("summary.json", payload)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
