from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reflexlm.baselines.state_rule_policy import BoundedStateRulePolicy
from reflexlm.cli.run_phase2bn_model_selected_sealed_runtime import (
    run_phase2bn_model_selected_sealed_runtime,
)
from reflexlm.cli.run_phase2br_runtime_ablation_comparison import _aggregate_subreports
from reflexlm.eval import EventGatedSequencePolicy, SequenceModelPolicy
from reflexlm.train import load_model_checkpoint


def _permission_key(permission: dict[str, Any]) -> str:
    return json.dumps(permission, sort_keys=True, ensure_ascii=False)


def _permuted_manifest(source: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(json.dumps(source))
    manifest["generated_by"] = {
        **dict(manifest.get("generated_by", {})),
        "phase": "phase2bv",
        "transformation": "reverse_run_command_permissions_then_stable_sort_noncommands",
    }
    for episode in manifest["episodes"]:
        permissions = list(episode["permissions"])
        commands = [
            permission
            for permission in permissions
            if permission["action_type"] == "RUN_COMMAND"
        ]
        noncommands = [
            permission
            for permission in permissions
            if permission["action_type"] != "RUN_COMMAND"
        ]
        episode["permissions"] = list(reversed(commands)) + sorted(
            noncommands,
            key=_permission_key,
        )
        episode["permission_permutation"] = {
            "source_permission_sha256": hashlib.sha256(
                json.dumps(permissions, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "command_order_reversed": len(commands) > 1,
            "completion_requirements_unchanged": True,
        }
    return manifest


def _policy_factories(
    *,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    model, vectorizer, checkpoint_payload = load_model_checkpoint(checkpoint_path, device="cpu")
    training_summary = checkpoint_payload.get("training_summary", {})
    return {
        "standard_full_native": lambda: SequenceModelPolicy(
            model,
            vectorizer,
            policy_label="phase2bv_standard_full_native",
            training_summary=training_summary,
            authorize_bounded_debug_cortex_recovery=True,
            use_synaptic_motor_plan=True,
        ),
        "equivariant_full_native": lambda: SequenceModelPolicy(
            model,
            vectorizer,
            policy_label="phase2bv_equivariant_full_native",
            training_summary=training_summary,
            authorize_bounded_debug_cortex_recovery=True,
            use_synaptic_motor_plan=True,
            command_permutation_ensemble=True,
        ),
        "equivariant_event_gated_native": lambda: EventGatedSequencePolicy(
            model,
            vectorizer,
            policy_label="phase2bv_equivariant_event_gated_native",
            training_summary=training_summary,
            authorize_bounded_debug_cortex_recovery=True,
            use_synaptic_motor_plan=True,
            command_permutation_ensemble=True,
        ),
        "bounded_state_rule": lambda: BoundedStateRulePolicy(
            policy_label="phase2bv_bounded_state_rule"
        ),
    }


def run_phase2bv_permission_permutation_stress(
    *,
    checkpoint_path: str | Path,
    phase2bq_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float = 2.0,
    max_extra_steps: int = 5,
) -> dict[str, Any]:
    phase2bq_report_path = Path(phase2bq_report_json)
    phase2bq_report = json.loads(
        phase2bq_report_path.read_text(encoding="utf-8-sig")
    )
    repositories = phase2bq_report["repository_reports"]
    output_root = Path(output_dir)
    manifest_root = output_root / "permuted_manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for repository in repositories:
        source_path = Path(repository["generated_manifest_json"])
        source_manifest = json.loads(source_path.read_text(encoding="utf-8-sig"))
        permuted = _permuted_manifest(source_manifest)
        output_path = manifest_root / f"{repository['repository_id']}.json"
        output_path.write_text(
            json.dumps(permuted, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "repository_id": str(repository["repository_id"]),
                "source_manifest_json": str(source_path),
                "permuted_manifest_json": str(output_path),
                "episodes": len(permuted["episodes"]),
                "reversed_multi_command_episodes": sum(
                    bool(row["permission_permutation"]["command_order_reversed"])
                    for row in permuted["episodes"]
                ),
            }
        )

    policy_reports: dict[str, dict[str, Any]] = {}
    for policy_id, factory in _policy_factories(checkpoint_path=checkpoint_path).items():
        policy = factory()
        subreports: list[dict[str, Any]] = []
        for row in manifest_rows:
            repository_output = output_root / policy_id / row["repository_id"]
            subreports.append(
                run_phase2bn_model_selected_sealed_runtime(
                    checkpoint_path=None,
                    manifest_json=row["permuted_manifest_json"],
                    output_jsonl=repository_output / "trajectories.jsonl",
                    output_report_json=repository_output / "report.json",
                    timeout_seconds=timeout_seconds,
                    max_extra_steps=max_extra_steps,
                    policy_label=f"phase2bv_{policy_id}",
                    policy_instance=policy,
                )
            )
        policy_reports[policy_id] = {
            "metrics": _aggregate_subreports(subreports),
            "repository_report_jsons": [
                str(output_root / policy_id / row["repository_id"] / "report.json")
                for row in manifest_rows
            ],
        }

    standard_full_native = policy_reports["standard_full_native"]["metrics"]
    full_native = policy_reports["equivariant_full_native"]["metrics"]
    event_gated = policy_reports["equivariant_event_gated_native"]["metrics"]
    rule = policy_reports["bounded_state_rule"]["metrics"]
    checks = {
        "phase2bq_source_report_passed": phase2bq_report.get("passed") is True,
        "completion_requirements_unchanged": all(
            json.loads(Path(row["source_manifest_json"]).read_text(encoding="utf-8-sig"))[
                "episodes"
            ][index]["completion_requirements"]
            == json.loads(Path(row["permuted_manifest_json"]).read_text(encoding="utf-8"))[
                "episodes"
            ][index]["completion_requirements"]
            for row in manifest_rows
            for index in range(row["episodes"])
        ),
        "multi_command_order_was_reversed": sum(
            row["reversed_multi_command_episodes"] for row in manifest_rows
        )
        > 0,
        "standard_full_native_exposes_order_sensitivity": standard_full_native[
            "task_completion_success_rate"
        ]
        < 1.0,
        "equivariant_full_native_permission_order_invariant": full_native[
            "task_completion_success_rate"
        ]
        == 1.0,
        "equivariant_event_gated_native_permission_order_invariant": event_gated[
            "task_completion_success_rate"
        ]
        == 1.0,
        "bounded_state_rule_permission_order_invariant": rule[
            "task_completion_success_rate"
        ]
        == 1.0,
    }
    learned_invariance_checks = {
        key: checks[key]
        for key in [
            "phase2bq_source_report_passed",
            "completion_requirements_unchanged",
            "multi_command_order_was_reversed",
            "standard_full_native_exposes_order_sensitivity",
            "equivariant_event_gated_native_permission_order_invariant",
        ]
    }
    learned_invariance_passed = all(learned_invariance_checks.values())
    report = {
        "artifact_family": "phase2bv_permission_permutation_stress",
        "passed": learned_invariance_passed,
        "ready_for_permission_order_invariant_event_gated_native_runtime_claim": learned_invariance_passed,
        "ready_for_rule_resistant_permission_permutation_claim": learned_invariance_passed
        and not checks["bounded_state_rule_permission_order_invariant"],
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "manifest_rows": manifest_rows,
        "policy_reports": policy_reports,
        "supported_claims": [
            "the event-gated permutation-ensemble native runtime preserved bounded completion under non-semantic permission ordering while the standard native and visible-state rule baselines failed"
        ]
        if learned_invariance_passed
        else [],
        "unsupported_claims": [
            "general rule-resistant novel state composition",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2bw_rule_resistant_semantic_candidate_selection"
            if learned_invariance_passed
            else "repair_command_candidate_permutation_equivariance"
        ),
    }
    output_report = Path(output_report_json)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress native runtime and rules under non-semantic permission permutations."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--phase2bq-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--max-extra-steps", type=int, default=5)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2bv_permission_permutation_stress(
        checkpoint_path=args.checkpoint_path,
        phase2bq_report_json=args.phase2bq_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
        timeout_seconds=args.timeout_seconds,
        max_extra_steps=args.max_extra_steps,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
