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
from reflexlm.cli.run_phase2by_learned_semantic_affordance import _training_groups
from reflexlm.eval import EventGatedSequencePolicy
from reflexlm.models.semantic_matcher import HashedDualEncoderSemanticMatcher
from reflexlm.train import load_model_checkpoint


OPEN_VOCABULARY_RECIPES = {
    "disk_full": {
        "failure": "No space left on device while writing cache",
        "correct": "disk",
        "distractors": ["certificate", "database"],
    },
    "certificate_expired": {
        "failure": "TLS certificate expired during handshake",
        "correct": "certificate",
        "distractors": ["memory", "config"],
    },
    "invalid_config": {
        "failure": "configuration syntax invalid near line four",
        "correct": "config",
        "distractors": ["database", "disk"],
    },
    "database_locked": {
        "failure": "database is locked by another process",
        "correct": "database",
        "distractors": ["certificate", "memory"],
    },
    "out_of_memory": {
        "failure": "out of memory while compiling module",
        "correct": "memory",
        "distractors": ["config", "disk"],
    },
}

OPEN_VOCABULARY_COMMANDS = {
    "disk": "free device disk space",
    "certificate": "renew expired TLS certificate",
    "config": "validate configuration syntax",
    "database": "release database lock",
    "memory": "reduce memory use",
}


def _command(candidate: str, marker: str) -> dict[str, Any]:
    return {
        "action_type": "RUN_COMMAND",
        "argv": [
            "<PYTHON>",
            "-c",
            f"print('{OPEN_VOCABULARY_COMMANDS[candidate]}: {marker}')",
        ],
        "expected_exit_code": 0,
    }


def _manifest(
    *,
    workspace_root: str | Path,
    repository_id: str,
    erased: bool,
) -> dict[str, Any]:
    episodes = []
    for recipe_id, recipe in OPEN_VOCABULARY_RECIPES.items():
        marker = f"phase2bz_{repository_id}_candidate"
        correct = str(recipe["correct"])
        order = [*recipe["distractors"], correct]
        commands = {candidate: _command(candidate, marker) for candidate in order}
        episodes.append(
            {
                "episode_id": f"phase2bz_{repository_id}_{recipe_id}_{'erased' if erased else 'visible'}",
                "task_type": "test_failure_reflex",
                "description": "Recover from an unseen visible failure using an allowlisted candidate.",
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
                        "stderr_delta": (
                            "process exited; unseen semantic details withheld"
                            if erased
                            else str(recipe["failure"])
                        ),
                        "stderr_unread": True,
                        "stderr_lines": 1,
                        "last_output_channel": "stderr",
                    },
                },
                "open_vocabulary_contract": {
                    "recipe_id": recipe_id,
                    "semantic_signal_erased": erased,
                    "candidate_order": order,
                    "correct_candidate": correct,
                    "correct_candidate_is_first": order[0] == correct,
                    "expected_sequence_exposed": False,
                },
            }
        )
    return {
        "workspace_root": str(Path(workspace_root).resolve()),
        "generated_by": {
            "phase": "phase2bz",
            "repository_id": repository_id,
            "semantic_signal_erased": erased,
        },
        "episodes": episodes,
    }


def _policy_factories(
    *,
    checkpoint_path: str | Path,
    lexical_matcher: HashedDualEncoderSemanticMatcher,
    dual_only_matcher: HashedDualEncoderSemanticMatcher,
) -> dict[str, Any]:
    model, vectorizer, checkpoint_payload = load_model_checkpoint(checkpoint_path, device="cpu")
    common = {
        "model": model,
        "vectorizer": vectorizer,
        "training_summary": checkpoint_payload.get("training_summary", {}),
        "authorize_bounded_debug_cortex_recovery": True,
        "use_synaptic_motor_plan": True,
        "command_permutation_ensemble": True,
    }
    return {
        "learned_lexical_residual_native": lambda: EventGatedSequencePolicy(
            policy_label="phase2bz_learned_lexical_residual_native",
            semantic_command_prior_weight=8.0,
            semantic_command_scorer=lexical_matcher,
            **common,
        ),
        "learned_dual_only_native": lambda: EventGatedSequencePolicy(
            policy_label="phase2bz_learned_dual_only_native",
            semantic_command_prior_weight=8.0,
            semantic_command_scorer=dual_only_matcher,
            **common,
        ),
        "ontology_semantic_native": lambda: EventGatedSequencePolicy(
            policy_label="phase2bz_ontology_semantic_native",
            semantic_command_prior_weight=1.0,
            **common,
        ),
        "equivariant_no_semantic_prior": lambda: EventGatedSequencePolicy(
            policy_label="phase2bz_equivariant_no_semantic_prior",
            **common,
        ),
        "bounded_state_rule": lambda: BoundedStateRulePolicy(
            policy_label="phase2bz_bounded_state_rule"
        ),
    }


def _run_suite(
    *,
    rows: list[dict[str, str]],
    checkpoint_path: str | Path,
    lexical_matcher: HashedDualEncoderSemanticMatcher,
    dual_only_matcher: HashedDualEncoderSemanticMatcher,
    output_root: Path,
    timeout_seconds: float,
) -> dict[str, dict[str, Any]]:
    reports = {}
    for policy_id, factory in _policy_factories(
        checkpoint_path=checkpoint_path,
        lexical_matcher=lexical_matcher,
        dual_only_matcher=dual_only_matcher,
    ).items():
        policy = factory()
        subreports = []
        for row in rows:
            repository_output = output_root / policy_id / row["repository_id"]
            subreports.append(
                run_phase2bn_model_selected_sealed_runtime(
                    checkpoint_path=None,
                    manifest_json=row["manifest_json"],
                    output_jsonl=repository_output / "trajectories.jsonl",
                    output_report_json=repository_output / "report.json",
                    timeout_seconds=timeout_seconds,
                    max_extra_steps=3,
                    policy_label=f"phase2bz_{policy_id}",
                    policy_instance=policy,
                )
            )
        reports[policy_id] = {
            "metrics": _aggregate_subreports(subreports),
            "repository_report_jsons": [
                str(output_root / policy_id / row["repository_id"] / "report.json")
                for row in rows
            ],
        }
    return reports


def run_phase2bz_open_vocabulary_transfer(
    *,
    checkpoint_path: str | Path,
    phase2bq_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    phase2bq_report = json.loads(Path(phase2bq_report_json).read_text(encoding="utf-8-sig"))
    output_root = Path(output_dir)
    base_matcher = HashedDualEncoderSemanticMatcher(bins=2048, embedding_dim=96, seed=17)
    base_matcher.fit(_training_groups(), epochs=2200, learning_rate=0.025)
    matcher_path = base_matcher.save(output_root / "closed_concept_matcher.pt")
    dual_only_matcher = HashedDualEncoderSemanticMatcher.load(matcher_path)
    lexical_matcher = HashedDualEncoderSemanticMatcher.load(matcher_path)
    lexical_matcher.lexical_residual_weight = 3.0
    suites: dict[str, list[dict[str, str]]] = {"visible": [], "erased": []}
    manifest_root = output_root / "manifests"
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
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            suites[suite_id].append(
                {"repository_id": repository_id, "manifest_json": str(manifest_path)}
            )
    visible_reports = _run_suite(
        rows=suites["visible"],
        checkpoint_path=checkpoint_path,
        lexical_matcher=lexical_matcher,
        dual_only_matcher=dual_only_matcher,
        output_root=output_root / "visible",
        timeout_seconds=timeout_seconds,
    )
    erased_reports = _run_suite(
        rows=suites["erased"],
        checkpoint_path=checkpoint_path,
        lexical_matcher=lexical_matcher,
        dual_only_matcher=dual_only_matcher,
        output_root=output_root / "erased",
        timeout_seconds=timeout_seconds,
    )
    lexical_visible = visible_reports["learned_lexical_residual_native"]["metrics"]
    lexical_erased = erased_reports["learned_lexical_residual_native"]["metrics"]
    dual_visible = visible_reports["learned_dual_only_native"]["metrics"]
    ontology_visible = visible_reports["ontology_semantic_native"]["metrics"]
    no_prior_visible = visible_reports["equivariant_no_semantic_prior"]["metrics"]
    rule_visible = visible_reports["bounded_state_rule"]["metrics"]
    training_text = " ".join(
        text
        for group in _training_groups().values()
        for values in group.values()
        for text in values
    ).lower()
    checks = {
        "phase2bq_source_report_passed": phase2bq_report.get("passed") is True,
        "open_vocabulary_concept_ids_absent_from_training_groups": all(
            concept not in _training_groups() for concept in OPEN_VOCABULARY_COMMANDS
        ),
        "open_vocabulary_failure_texts_not_copied_into_training": all(
            str(recipe["failure"]).lower() not in training_text
            for recipe in OPEN_VOCABULARY_RECIPES.values()
        ),
        "lexical_residual_native_solves_unseen_visible_tasks": lexical_visible[
            "task_completion_success_rate"
        ]
        == 1.0,
        "semantic_signal_erasure_reduces_lexical_transfer": lexical_erased[
            "task_completion_success_rate"
        ]
        < lexical_visible["task_completion_success_rate"],
        "lexical_residual_outperforms_dual_only": lexical_visible[
            "task_completion_success_rate"
        ]
        > dual_visible["task_completion_success_rate"],
        "lexical_residual_outperforms_ontology": lexical_visible[
            "task_completion_success_rate"
        ]
        > ontology_visible["task_completion_success_rate"],
        "lexical_residual_outperforms_no_prior": lexical_visible[
            "task_completion_success_rate"
        ]
        > no_prior_visible["task_completion_success_rate"],
        "lexical_residual_outperforms_state_rule": lexical_visible[
            "task_completion_success_rate"
        ]
        > rule_visible["task_completion_success_rate"],
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2bz_open_vocabulary_transfer",
        "passed": passed,
        "ready_for_bounded_open_vocabulary_lexical_transfer_claim": passed,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checkpoint_path": str(checkpoint_path),
        "phase2bq_report_json": str(phase2bq_report_json),
        "checks": checks,
        "matcher_path": str(matcher_path),
        "lexical_matcher_metadata": lexical_matcher.metadata(),
        "visible_policy_reports": visible_reports,
        "erased_policy_reports": erased_reports,
        "comparison": {
            "lexical_visible_completion": lexical_visible["task_completion_success_rate"],
            "lexical_erased_completion": lexical_erased["task_completion_success_rate"],
            "dual_visible_completion": dual_visible["task_completion_success_rate"],
            "ontology_visible_completion": ontology_visible["task_completion_success_rate"],
            "no_prior_visible_completion": no_prior_visible["task_completion_success_rate"],
            "rule_visible_completion": rule_visible["task_completion_success_rate"],
        },
        "supported_claims": [
            "a generic lexical residual enabled bounded transfer to unseen failure concepts without runtime ontology lookup and degraded when semantic receptor evidence was erased"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "paraphrase-robust open-vocabulary semantics without lexical overlap",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2ca_open_vocabulary_no_lexical_overlap_transfer"
            if passed
            else "repair_phase2bz_open_vocabulary_transfer"
        ),
    }
    output_path = Path(output_report_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate bounded transfer to unseen semantic recovery concepts."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--phase2bq-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2bz_open_vocabulary_transfer(
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
