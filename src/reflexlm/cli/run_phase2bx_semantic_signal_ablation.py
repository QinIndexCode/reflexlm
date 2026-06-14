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


RECIPES: dict[str, dict[str, Any]] = {
    "dependency_paraphrase": {
        "failure": "Import failed because package beta_widget is unavailable",
        "correct": "dependency",
        "distractors": ["snapshot", "permission"],
    },
    "snapshot_paraphrase": {
        "failure": "golden output differs from stored reference in renderer",
        "correct": "snapshot",
        "distractors": ["dependency", "assertion"],
    },
    "assertion_paraphrase": {
        "failure": "expected 4 but received 5 during test verification",
        "correct": "assertion",
        "distractors": ["snapshot", "path"],
    },
    "port_occupied": {
        "failure": "Address already in use on TCP port 8123",
        "correct": "port",
        "distractors": ["permission", "path"],
    },
    "permission_denied": {
        "failure": "Access is denied while writing config file",
        "correct": "permission",
        "distractors": ["dependency", "port"],
    },
    "missing_path": {
        "failure": "FileNotFoundError: target directory does not exist",
        "correct": "path",
        "distractors": ["snapshot", "permission"],
    },
    "dependency_and_path": {
        "failure": "Import failed and target directory does not exist",
        "correct": "dependency_path",
        "distractors": ["dependency", "path"],
    },
    "port_and_permission": {
        "failure": "Address already in use and access is denied during restart",
        "correct": "port_permission",
        "distractors": ["port", "permission"],
    },
}

COMMAND_TEXT = {
    "dependency": "install package dependency recovery",
    "snapshot": "regenerate expected output update snapshot recovery",
    "assertion": "pytest targeted test verification recovery",
    "port": "release occupied TCP port recovery",
    "permission": "repair file permissions recovery",
    "path": "create missing directory ensure path recovery",
    "dependency_path": "install package dependency and create missing directory",
    "port_permission": "release occupied TCP port and repair file permissions",
}


def _command(candidate: str, marker: str) -> dict[str, Any]:
    return {
        "action_type": "RUN_COMMAND",
        "argv": ["<PYTHON>", "-c", f"print('{COMMAND_TEXT[candidate]}: {marker}')"],
        "expected_exit_code": 0,
    }


def _episode(*, repository_id: str, recipe_id: str, erased: bool) -> dict[str, Any]:
    recipe = RECIPES[recipe_id]
    marker = f"phase2bx_{repository_id}_candidate"
    correct = str(recipe["correct"])
    order = [*recipe["distractors"], correct]
    commands = {candidate: _command(candidate, marker) for candidate in order}
    failure_text = (
        "process exited with a bounded failure; semantic details withheld"
        if erased
        else str(recipe["failure"])
    )
    return {
        "episode_id": f"phase2bx_{repository_id}_{recipe_id}_{'erased' if erased else 'visible'}",
        "task_type": "test_failure_reflex",
        "description": "Recover from the visible terminal failure using an allowlisted candidate.",
        "permissions": [
            *(commands[candidate] for candidate in order),
            {"action_type": "READ_STDERR"},
            {"action_type": "READ_STDOUT"},
            {"action_type": "DONE"},
        ],
        "completion_requirements": [
            {"action_type": "READ_STDERR"},
            commands[correct],
            {"action_type": "READ_STDOUT"},
        ],
        "requires_failure": True,
        "max_steps": 6,
        "initial_state": {
            "process": {"status": "exited", "exit_code": 1},
            "terminal": {
                "stderr_delta": failure_text,
                "stderr_unread": True,
                "stderr_lines": 1,
                "last_output_channel": "stderr",
            },
        },
        "semantic_ablation_contract": {
            "recipe_id": recipe_id,
            "semantic_signal_erased": erased,
            "candidate_order": order,
            "correct_candidate": correct,
            "correct_candidate_is_first": order[0] == correct,
            "expected_sequence_exposed": False,
        },
    }


def _manifest(
    *,
    workspace_root: str | Path,
    repository_id: str,
    erased: bool,
) -> dict[str, Any]:
    return {
        "workspace_root": str(Path(workspace_root).resolve()),
        "repetitions_per_episode": 1,
        "generated_by": {
            "phase": "phase2bx",
            "repository_id": repository_id,
            "semantic_signal_erased": erased,
        },
        "episodes": [
            _episode(repository_id=repository_id, recipe_id=recipe_id, erased=erased)
            for recipe_id in RECIPES
        ],
    }


def _policy_factories(*, checkpoint_path: str | Path) -> dict[str, Any]:
    model, vectorizer, checkpoint_payload = load_model_checkpoint(checkpoint_path, device="cpu")
    training_summary = checkpoint_payload.get("training_summary", {})
    return {
        "semantic_event_gated_native": lambda: EventGatedSequencePolicy(
            model,
            vectorizer,
            policy_label="phase2bx_semantic_event_gated_native",
            training_summary=training_summary,
            authorize_bounded_debug_cortex_recovery=True,
            use_synaptic_motor_plan=True,
            command_permutation_ensemble=True,
            semantic_command_prior_weight=1.0,
        ),
        "equivariant_no_semantic_prior": lambda: EventGatedSequencePolicy(
            model,
            vectorizer,
            policy_label="phase2bx_equivariant_no_semantic_prior",
            training_summary=training_summary,
            authorize_bounded_debug_cortex_recovery=True,
            use_synaptic_motor_plan=True,
            command_permutation_ensemble=True,
        ),
        "bounded_state_rule": lambda: BoundedStateRulePolicy(
            policy_label="phase2bx_bounded_state_rule"
        ),
    }


def _run_policy_suite(
    *,
    manifest_rows: list[dict[str, Any]],
    checkpoint_path: str | Path,
    output_root: Path,
    timeout_seconds: float,
) -> dict[str, dict[str, Any]]:
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
                    policy_label=f"phase2bx_{policy_id}",
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
    return policy_reports


def run_phase2bx_semantic_signal_ablation(
    *,
    checkpoint_path: str | Path,
    phase2bq_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    phase2bq_report = json.loads(Path(phase2bq_report_json).read_text(encoding="utf-8-sig"))
    output_root = Path(output_dir)
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    suites: dict[str, list[dict[str, Any]]] = {"visible": [], "erased": []}
    for suite_id, erased in [("visible", False), ("erased", True)]:
        for repository in phase2bq_report["repository_reports"]:
            repository_id = str(repository["repository_id"])
            manifest = _manifest(
                workspace_root=repository["provenance"]["git_root"],
                repository_id=repository_id,
                erased=erased,
            )
            manifest_path = manifest_root / suite_id / f"{repository_id}.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            suites[suite_id].append(
                {
                    "repository_id": repository_id,
                    "manifest_json": str(manifest_path),
                    "episode_templates": len(manifest["episodes"]),
                }
            )

    visible_reports = _run_policy_suite(
        manifest_rows=suites["visible"],
        checkpoint_path=checkpoint_path,
        output_root=output_root / "visible",
        timeout_seconds=timeout_seconds,
    )
    erased_reports = _run_policy_suite(
        manifest_rows=suites["erased"],
        checkpoint_path=checkpoint_path,
        output_root=output_root / "erased",
        timeout_seconds=timeout_seconds,
    )
    semantic_visible = visible_reports["semantic_event_gated_native"]["metrics"]
    semantic_erased = erased_reports["semantic_event_gated_native"]["metrics"]
    no_prior_visible = visible_reports["equivariant_no_semantic_prior"]["metrics"]
    rule_visible = visible_reports["bounded_state_rule"]["metrics"]
    generated_manifests = [
        json.loads(Path(row["manifest_json"]).read_text(encoding="utf-8"))
        for rows in suites.values()
        for row in rows
    ]
    checks = {
        "phase2bq_source_report_passed": phase2bq_report.get("passed") is True,
        "generated_manifests_have_no_expected_sequence_or_steps": all(
            "expected_sequence" not in episode and "steps" not in episode
            for manifest in generated_manifests
            for episode in manifest["episodes"]
        ),
        "all_correct_candidates_are_nonfirst": all(
            episode["semantic_ablation_contract"]["correct_candidate_is_first"] is False
            for manifest in generated_manifests
            for episode in manifest["episodes"]
        ),
        "semantic_native_solves_visible_paraphrase_and_composition_tasks": semantic_visible[
            "task_completion_success_rate"
        ]
        == 1.0,
        "semantic_signal_erasure_reduces_completion": semantic_erased[
            "task_completion_success_rate"
        ]
        < semantic_visible["task_completion_success_rate"],
        "semantic_prior_improves_over_no_prior_on_visible_tasks": semantic_visible[
            "task_completion_success_rate"
        ]
        > no_prior_visible["task_completion_success_rate"],
        "semantic_prior_improves_over_state_rule_on_visible_tasks": semantic_visible[
            "task_completion_success_rate"
        ]
        > rule_visible["task_completion_success_rate"],
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2bx_semantic_signal_ablation",
        "passed": passed,
        "ready_for_bounded_compositional_semantic_signal_claim": passed,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checkpoint_path": str(Path(checkpoint_path)),
        "checks": checks,
        "manifest_rows": suites,
        "visible_policy_reports": visible_reports,
        "erased_policy_reports": erased_reports,
        "comparison": {
            "semantic_visible_completion": semantic_visible["task_completion_success_rate"],
            "semantic_erased_completion": semantic_erased["task_completion_success_rate"],
            "no_prior_visible_completion": no_prior_visible["task_completion_success_rate"],
            "rule_visible_completion": rule_visible["task_completion_success_rate"],
        },
        "supported_claims": [
            "bounded semantic affordance scoring handles paraphrased and compositional visible failures and degrades when the semantic receptor signal is erased"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "learned semantic affordance discovery without a bounded ontology",
            "unrestricted semantic reasoning",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2by_learned_semantic_affordance_without_handwritten_ontology"
            if passed
            else "repair_phase2bx_semantic_signal_ablation"
        ),
    }
    output_path = Path(output_report_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate bounded semantic receptor signals under paraphrase and composition."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--phase2bq-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2bx_semantic_signal_ablation(
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
