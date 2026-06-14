from __future__ import annotations

import argparse
import json

from reflexlm.llm.native_cortex import OPEN_REPAIR_CAPABILITY_NAMES
from reflexlm.llm.native_head_policy import MODEL_LOAD_STRATEGIES
from reflexlm.llm.native_nervous_package import write_native_nervous_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Phase2D native nervous policy package manifest.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model-name", required=True)
    parser.add_argument("--native-head-path", required=True)
    parser.add_argument("--low-level-checkpoint-path", required=True)
    parser.add_argument("--quantization", default="4bit")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--nsi-device", default="cpu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-load-strategy",
        choices=list(MODEL_LOAD_STRATEGIES),
        default="auto",
    )
    parser.add_argument("--offload-state-dict", action="store_true")
    parser.add_argument("--policy-label", default="phase2d_native_nervous_package")
    parser.add_argument("--zero-nsi-latent", action="store_true")
    parser.add_argument("--disable-continuation-cache", action="store_true")
    parser.add_argument("--disable-native-head-calls", action="store_true")
    parser.add_argument(
        "--disable-command-candidate-feature-group",
        action="append",
        default=[],
        help="Candidate feature group to zero at runtime; may be repeated or comma-separated.",
    )
    parser.add_argument(
        "--open-repair-capability",
        action="append",
        default=[],
        choices=list(OPEN_REPAIR_CAPABILITY_NAMES),
        help="Declare a trained open-repair runtime capability; may be repeated.",
    )
    parser.add_argument(
        "--patch-proposal-strategy",
        choices=[
            "none",
            "recorded_candidate_selection",
            "symbolic_runtime_generator",
            "learned_bounded_candidate",
        ],
        default="none",
        help="Declare the provenance strategy for patch proposal evidence.",
    )
    parser.add_argument(
        "--learned-patch-generation-enabled",
        action="store_true",
        help="Declare that this package has trained learned bounded patch generation.",
    )
    parser.add_argument(
        "--patch-candidate-schema-version",
        help="Schema version for learned bounded patch candidates.",
    )
    args = parser.parse_args()
    open_repair_capabilities = {
        name: name in set(args.open_repair_capability)
        for name in OPEN_REPAIR_CAPABILITY_NAMES
    }

    manifest = write_native_nervous_package(
        args.output_dir,
        base_model_name=args.base_model_name,
        native_head_path=args.native_head_path,
        low_level_checkpoint_path=args.low_level_checkpoint_path,
        quantization=args.quantization,
        max_length=args.max_length,
        nsi_device=args.nsi_device,
        device=args.device,
        model_load_strategy=args.model_load_strategy,
        offload_state_dict=args.offload_state_dict,
        policy_label=args.policy_label,
        zero_nsi_latent=args.zero_nsi_latent,
        continuation_cache_enabled=not args.disable_continuation_cache,
        native_head_calls_enabled=not args.disable_native_head_calls,
        disabled_command_candidate_feature_groups=args.disable_command_candidate_feature_group,
        open_repair_capabilities=open_repair_capabilities,
        patch_proposal_strategy=args.patch_proposal_strategy,
        learned_patch_generation_enabled=args.learned_patch_generation_enabled,
        patch_candidate_schema_version=args.patch_candidate_schema_version,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
