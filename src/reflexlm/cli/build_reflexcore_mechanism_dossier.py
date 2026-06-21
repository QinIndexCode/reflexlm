from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.mechanism_dossier import (
    ReflexCoreMechanismDossierConfig,
    build_reflexcore_mechanism_dossier,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a bounded ReflexCore V0 mechanism evidence dossier.",
    )
    parser.add_argument("--accepted-rollup-json", required=True)
    parser.add_argument("--sensory-ablation-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--architecture-audit-json")
    parser.add_argument("--runtime-evidence-audit-json")
    parser.add_argument("--negative-control-json", action="append", default=[])
    parser.add_argument("--min-parameter-count", type=int, default=20_000_000)
    parser.add_argument("--max-parameter-count", type=int, default=100_000_000)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-profile-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-raw-action-accuracy", type=float, default=0.85)
    parser.add_argument("--min-safety-gated-action-accuracy", type=float, default=0.70)
    parser.add_argument("--min-offline-action-margin", type=float, default=0.20)
    parser.add_argument("--min-closed-loop-success-rate", type=float, default=0.60)
    parser.add_argument("--min-closed-loop-margin", type=float, default=0.30)
    parser.add_argument("--min-real-sandbox-success-rate", type=float, default=1.0)
    parser.add_argument("--min-real-sandbox-margin", type=float, default=0.50)
    parser.add_argument("--min-next-state-relative-improvement", type=float, default=0.30)
    parser.add_argument(
        "--min-prediction-error-relative-improvement",
        type=float,
        default=0.30,
    )
    parser.add_argument("--required-ablation-mode", action="append", default=[])
    parser.add_argument("--min-sensory-action-drop", type=float, default=0.50)
    parser.add_argument("--min-sensory-world-drop", type=float, default=1.0)
    parser.add_argument("--min-sensory-rows", type=int, default=9)
    parser.add_argument("--required-seed", action="append", type=int, default=[])
    parser.add_argument("--required-profile", action="append", default=[])
    args = parser.parse_args()

    report = build_reflexcore_mechanism_dossier(
        ReflexCoreMechanismDossierConfig(
            accepted_rollup_json=Path(args.accepted_rollup_json),
            sensory_ablation_json=Path(args.sensory_ablation_json),
            output_json=Path(args.output_json) if args.output_json else None,
            architecture_audit_json=(
                Path(args.architecture_audit_json)
                if args.architecture_audit_json
                else None
            ),
            runtime_evidence_audit_json=(
                Path(args.runtime_evidence_audit_json)
                if args.runtime_evidence_audit_json
                else None
            ),
            negative_control_jsons=tuple(
                Path(path) for path in args.negative_control_json
            ),
            min_parameter_count=args.min_parameter_count,
            max_parameter_count=args.max_parameter_count,
            min_pass_rate=args.min_pass_rate,
            min_profile_pass_rate=args.min_profile_pass_rate,
            min_raw_action_accuracy=args.min_raw_action_accuracy,
            min_safety_gated_action_accuracy=args.min_safety_gated_action_accuracy,
            min_offline_action_margin=args.min_offline_action_margin,
            min_closed_loop_success_rate=args.min_closed_loop_success_rate,
            min_closed_loop_margin=args.min_closed_loop_margin,
            min_real_sandbox_success_rate=args.min_real_sandbox_success_rate,
            min_real_sandbox_margin=args.min_real_sandbox_margin,
            min_next_state_relative_improvement=args.min_next_state_relative_improvement,
            min_prediction_error_relative_improvement=(
                args.min_prediction_error_relative_improvement
            ),
            required_ablation_modes=tuple(args.required_ablation_mode)
            if args.required_ablation_mode
            else ("zero_numeric",),
            min_sensory_action_drop=args.min_sensory_action_drop,
            min_sensory_world_drop=args.min_sensory_world_drop,
            min_sensory_rows=args.min_sensory_rows,
            required_seeds=tuple(args.required_seed)
            if args.required_seed
            else (13, 17, 23),
            required_profiles=tuple(args.required_profile)
            if args.required_profile
            else ("default", "hard", "wide_ood"),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
