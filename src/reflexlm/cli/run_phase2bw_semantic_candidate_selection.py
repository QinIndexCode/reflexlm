from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.baselines.state_rule_policy import BoundedStateRulePolicy
from reflexlm.cli.run_phase2bn_model_selected_sealed_runtime import (
    run_phase2bn_model_selected_sealed_runtime,
)
from reflexlm.cli.run_phase2br_runtime_ablation_comparison import _aggregate_subreports
from reflexlm.eval import EventGatedSequencePolicy
from reflexlm.train import load_model_checkpoint


FAILURE_RECIPES = {
    "dependency": {
        "failure": "ModuleNotFoundError: No module named 'phase2bw_dependency'",
        "correct": "dependency",
    },
    "snapshot": {
        "failure": "snapshot mismatch: stored rendering requires update",
        "correct": "snapshot",
    },
    "assertion": {
        "failure": "AssertionError: assertion failure in bounded semantic candidate test",
        "correct": "assertion",
    },
}


def _command(candidate: str, marker: str) -> dict[str, Any]:
    semantic_text = {
        "dependency": "pip install dependency recovery",
        "snapshot": "snapshot update recovery",
        "assertion": "pytest assertion recovery",
    }[candidate]
    return {
        "action_type": "RUN_COMMAND",
        "argv": [
            "<PYTHON>",
            "-c",
            f"print('{semantic_text}: {marker}')",
        ],
        "expected_exit_code": 0,
    }


def _semantic_manifest(
    *,
    workspace_root: str | Path,
    repository_id: str,
) -> dict[str, Any]:
    candidate_orders = {
        "dependency": ["snapshot", "assertion", "dependency"],
        "snapshot": ["dependency", "snapshot", "assertion"],
        "assertion": ["snapshot", "dependency", "assertion"],
    }
    episodes: list[dict[str, Any]] = []
    for failure_kind, recipe in FAILURE_RECIPES.items():
        episode_id = f"phase2bw_{repository_id}_{failure_kind}"
        marker = f"phase2bw_{repository_id}_candidate"
        commands = {
            candidate: _command(candidate, marker)
            for candidate in ["dependency", "snapshot", "assertion"]
        }
        order = candidate_orders[failure_kind]
        correct = commands[str(recipe["correct"])]
        episodes.append(
            {
                "episode_id": episode_id,
                "task_type": "test_failure_reflex",
                "description": (
                    "Select the allowlisted recovery candidate that semantically matches "
                    f"this visible failure: {recipe['failure']}"
                ),
                "permissions": [
                    *(commands[candidate] for candidate in order),
                    {"action_type": "READ_STDERR"},
                    {"action_type": "READ_STDOUT"},
                    {"action_type": "DONE"},
                ],
                "completion_requirements": [
                    {"action_type": "READ_STDERR"},
                    correct,
                    {"action_type": "READ_STDOUT"},
                ],
                "requires_failure": True,
                "max_steps": 6,
                "repetitions": 2,
                "initial_state": {
                    "process": {"status": "exited", "exit_code": 1},
                    "terminal": {
                        "stderr_delta": str(recipe["failure"]),
                        "stderr_unread": True,
                        "stderr_lines": 1,
                        "last_output_channel": "stderr",
                    },
                },
                "semantic_candidate_contract": {
                    "failure_kind": failure_kind,
                    "candidate_order": order,
                    "correct_candidate": recipe["correct"],
                    "correct_candidate_is_first": order[0] == recipe["correct"],
                    "expected_sequence_exposed": False,
                },
            }
        )
    return {
        "workspace_root": str(Path(workspace_root).resolve()),
        "repetitions_per_episode": 1,
        "generated_by": {
            "phase": "phase2bw",
            "repository_id": repository_id,
            "transformation": "visible_failure_semantic_candidate_selection",
        },
        "episodes": episodes,
    }


def _policy_factories(
    *,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    model, vectorizer, checkpoint_payload = load_model_checkpoint(checkpoint_path, device="cpu")
    training_summary = checkpoint_payload.get("training_summary", {})
    return {
        "standard_event_gated_native": lambda: EventGatedSequencePolicy(
            model,
            vectorizer,
            policy_label="phase2bw_standard_event_gated_native",
            training_summary=training_summary,
            authorize_bounded_debug_cortex_recovery=True,
            use_synaptic_motor_plan=True,
        ),
        "equivariant_event_gated_native": lambda: EventGatedSequencePolicy(
            model,
            vectorizer,
            policy_label="phase2bw_equivariant_event_gated_native",
            training_summary=training_summary,
            authorize_bounded_debug_cortex_recovery=True,
            use_synaptic_motor_plan=True,
            command_permutation_ensemble=True,
            semantic_command_prior_weight=1.0,
        ),
        "equivariant_event_gated_no_semantic_prior": lambda: EventGatedSequencePolicy(
            model,
            vectorizer,
            policy_label="phase2bw_equivariant_event_gated_no_semantic_prior",
            training_summary=training_summary,
            authorize_bounded_debug_cortex_recovery=True,
            use_synaptic_motor_plan=True,
            command_permutation_ensemble=True,
            semantic_command_prior_weight=0.0,
        ),
        "bounded_state_rule": lambda: BoundedStateRulePolicy(
            policy_label="phase2bw_bounded_state_rule"
        ),
    }


def run_phase2bw_semantic_candidate_selection(
    *,
    checkpoint_path: str | Path,
    phase2bq_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    phase2bq_report = json.loads(
        Path(phase2bq_report_json).read_text(encoding="utf-8-sig")
    )
    output_root = Path(output_dir)
    manifest_root = output_root / "semantic_manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for repository in phase2bq_report["repository_reports"]:
        repository_id = str(repository["repository_id"])
        manifest = _semantic_manifest(
            workspace_root=repository["provenance"]["git_root"],
            repository_id=repository_id,
        )
        manifest_path = manifest_root / f"{repository_id}.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "repository_id": repository_id,
                "manifest_json": str(manifest_path),
                "episode_templates": len(manifest["episodes"]),
                "episodes": sum(
                    int(episode.get("repetitions", 1)) for episode in manifest["episodes"]
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
                    manifest_json=row["manifest_json"],
                    output_jsonl=repository_output / "trajectories.jsonl",
                    output_report_json=repository_output / "report.json",
                    timeout_seconds=timeout_seconds,
                    max_extra_steps=3,
                    policy_label=f"phase2bw_{policy_id}",
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

    standard = policy_reports["standard_event_gated_native"]["metrics"]
    equivariant = policy_reports["equivariant_event_gated_native"]["metrics"]
    equivariant_no_prior = policy_reports[
        "equivariant_event_gated_no_semantic_prior"
    ]["metrics"]
    rule = policy_reports["bounded_state_rule"]["metrics"]
    generated_manifests = [
        json.loads(Path(row["manifest_json"]).read_text(encoding="utf-8"))
        for row in manifest_rows
    ]
    checks = {
        "phase2bq_source_report_passed": phase2bq_report.get("passed") is True,
        "generated_manifests_have_no_expected_sequence_or_steps": all(
            "expected_sequence" not in episode and "steps" not in episode
            for manifest in generated_manifests
            for episode in manifest["episodes"]
        ),
        "all_correct_candidates_are_nonfirst": all(
            episode["semantic_candidate_contract"]["correct_candidate_is_first"] is False
            for manifest in generated_manifests
            for episode in manifest["episodes"]
        ),
        "all_episodes_start_from_visible_failure_state": all(
            episode["initial_state"]["process"]["exit_code"] != 0
            and episode["initial_state"]["terminal"]["stderr_unread"] is True
            for manifest in generated_manifests
            for episode in manifest["episodes"]
        ),
        "equivariant_event_gated_native_completed_all_tasks": equivariant[
            "task_completion_success_rate"
        ]
        == 1.0,
        "equivariant_event_gated_native_recovered_all_failures": equivariant[
            "failure_recovery_success_rate"
        ]
        == 1.0,
        "bounded_state_rule_failed_semantic_selection": rule[
            "task_completion_success_rate"
        ]
        < equivariant["task_completion_success_rate"],
        "semantic_prior_improves_equivariant_completion": equivariant[
            "task_completion_success_rate"
        ]
        > equivariant_no_prior["task_completion_success_rate"],
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2bw_semantic_candidate_selection",
        "passed": passed,
        "ready_for_bounded_rule_resistant_semantic_candidate_selection_claim": passed,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checkpoint_path": str(Path(checkpoint_path)),
        "checks": checks,
        "manifest_rows": manifest_rows,
        "policy_reports": policy_reports,
        "comparison": {
            "equivariant_vs_rule_completion_rate_delta": equivariant[
                "task_completion_success_rate"
            ]
            - rule["task_completion_success_rate"],
            "equivariant_vs_standard_completion_rate_delta": equivariant[
                "task_completion_success_rate"
            ]
            - standard["task_completion_success_rate"],
            "semantic_prior_completion_rate_delta": equivariant[
                "task_completion_success_rate"
            ]
            - equivariant_no_prior["task_completion_success_rate"],
        },
        "supported_claims": [
            "the bounded event-gated permutation-ensemble native runtime selected semantically matching recovery candidates from visible failure state while the position-based state rule failed"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "unrestricted semantic reasoning",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2bx_semantic_signal_ablation_and_novel_failure_composition"
            if passed
            else "repair_phase2bw_semantic_candidate_selection"
        ),
    }
    output_path = Path(output_report_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare bounded semantic recovery selection against position rules."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--phase2bq-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2bw_semantic_candidate_selection(
        checkpoint_path=args.checkpoint_path,
        phase2bq_report_json=args.phase2bq_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
