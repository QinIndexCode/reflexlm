from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_phase2m_v2_boundary() -> dict[str, Any]:
    return {
        "artifact_family": "phase2m_v2_design_boundary",
        "phase": "Phase2M-v2",
        "center_claim": (
            "Test whether the bounded native nervous-interface mechanism generalizes "
            "to preregistered external repository traces without sealed-v3 feedback."
        ),
        "supported_before_phase2m_v2": [
            "native-head/debug-stage semantic-required mechanism on prior bounded evidence",
            "NSI latent contribution versus no-NSI on sealed v3",
            "Phase2L non-sealed continuation-memory counterfactual support",
        ],
        "not_supported_before_phase2m_v2": [
            "open-ended debugging or production autonomy",
            "sealed-v3 continuation-cache necessity",
            "full package beats native-head-only on sealed v3",
            "top-tier general mechanism claim across external repos, seeds, and model families",
        ],
        "phase2m_v1_boundary": {
            "synthetic_safe_split_use": "plumbing_smoke_only",
            "claim_bearing_training_allowed": False,
            "reasons": [
                "synthetic-only source kind",
                "declared baselines rather than code-measured baselines",
                "direct candidate slot markers in visible evidence",
            ],
        },
        "phase2m_v2_training_allowed_only_if": [
            "data health passes with no sealed references or hidden/gold hints",
            "candidate slot markers are absent from visible text and candidate commands",
            "baseline predictions match code-measured evaluators",
            "repo-disjoint holdout and split hashes are present",
            "design maturity reports ready_for_claim_bearing_training=true",
        ],
        "sealed_policy": {
            "sealed_v3_is_final_eval_only": True,
            "sealed_failures_used_for_training_sampling_or_tuning": False,
            "sealed_failures_used_for_nonsealed_profile_design_feedback": False,
        },
        "monitoring_policy": {
            "codex_automation_allowed": False,
            "heartbeat_control_source_allowed": False,
            "training_checks_only_on_user_instruction_or_explicit_interval": True,
            "required_check_inputs": [
                "process list",
                "nvidia-smi",
                "latest progress JSON",
                "training logs",
                "training summary",
                "postflight or gate artifact",
            ],
        },
        "claim_upgrade_policy": {
            "bounded_claim_until_all_gates_pass": True,
            "requires_external_public_holdout": True,
            "requires_multi_seed_reproduction": True,
            "requires_multi_model_reproduction": True,
        },
    }


def render_boundary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase2M-v2 Design Boundary",
        "",
        f"- Center claim: {report['center_claim']}",
        "- Phase2M-v1 synthetic-safe split remains plumbing-smoke only.",
        "- Claim-bearing training is allowed only after data health, pretrain, and design maturity gates all pass.",
        "- Sealed v3 remains final evaluation-only and cannot influence non-sealed data design.",
        "- Monitoring uses user-instruction intervals only; no Codex automation is created.",
        "",
        "## Supported Before Phase2M-v2",
    ]
    lines.extend(f"- {item}" for item in report["supported_before_phase2m_v2"])
    lines.extend(["", "## Not Supported Before Phase2M-v2"])
    lines.extend(f"- {item}" for item in report["not_supported_before_phase2m_v2"])
    lines.extend(["", "## Training Preconditions"])
    lines.extend(f"- {item}" for item in report["phase2m_v2_training_allowed_only_if"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Phase2M-v2 design boundary artifacts.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    report = build_phase2m_v2_boundary()
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(render_boundary_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
