from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_phase2r_dynamic_boundary() -> dict[str, Any]:
    return {
        "artifact_family": "phase2r_dynamic_public_trace_boundary",
        "phase": "Phase2R",
        "center_claim": (
            "Test whether the bounded native nervous-interface mechanism survives "
            "dynamic public-repository pytest execution traces, after Phase2Q "
            "public/read-only static trace breadth passed."
        ),
        "supported_before_phase2r": [
            "bounded semantic-required Debug Cortex / NSI mechanism on Phase2Q public read-only traces",
            "sealed-v3 transfer for the Phase2Q 7B package under the fixed semantic-required benchmark",
        ],
        "not_supported_before_phase2r": [
            "open-ended dynamic real-repo debugging or repair",
            "production autonomy",
            "unrestricted shell/tool use",
            "independent external reproduction",
            "epoch-making architecture status",
        ],
        "phase2r_training_allowed_only_if": [
            "collector uses isolated dynamic pytest subprocess execution",
            "source repositories remain read-only and are not mutated by collection",
            "no sealed-v3 artifacts, failures, or scores are used for training, sampling, or design feedback",
            "train/val/holdout repos are disjoint and public",
            "every row has dynamic execution evidence",
            "candidate slot markers and behavior-summary shortcuts are absent from visible evidence",
            "source-overlap, native-head-only, prompt-only, continuation-only, and ReAct baselines are code-measured and below threshold",
        ],
        "sealed_policy": {
            "sealed_v3_is_final_eval_only": True,
            "sealed_failures_used_for_training_sampling_or_tuning": False,
            "sealed_failures_used_for_nonsealed_profile_design_feedback": False,
        },
        "claim_upgrade_policy": {
            "bounded_claim_until_dynamic_nonsealed_and_holdout_pass": True,
            "production_autonomy_requires_live_repair_benchmarks_and_safety_gates": True,
            "epoch_making_architecture_requires_independent_external_reproduction": True,
        },
    }


def render_boundary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase2R Dynamic Public Trace Boundary",
        "",
        f"- Center claim: {report['center_claim']}",
        "- Phase2R is a non-sealed dynamic execution pressure benchmark, not a sealed-v3 tuning loop.",
        "- Passing Phase2R may strengthen bounded dynamic-trace evidence; it still cannot prove production autonomy by itself.",
        "",
        "## Supported Before Phase2R",
    ]
    lines.extend(f"- {item}" for item in report["supported_before_phase2r"])
    lines.extend(["", "## Not Supported Before Phase2R"])
    lines.extend(f"- {item}" for item in report["not_supported_before_phase2r"])
    lines.extend(["", "## Training Preconditions"])
    lines.extend(f"- {item}" for item in report["phase2r_training_allowed_only_if"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Phase2R dynamic trace boundary artifacts.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    report = build_phase2r_dynamic_boundary()
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(render_boundary_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
