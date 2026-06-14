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
from reflexlm.models.semantic_matcher import (
    CausalLMConditionalSemanticMatcher,
    HashedDualEncoderSemanticMatcher,
    _semantic_words,
)
from reflexlm.train import load_model_checkpoint


NO_LEXICAL_OVERLAP_RECIPES = {
    "capacity_exhausted": {
        "failure": "storage volume has exhausted all remaining capacity",
        "correct": "cache",
        "distractors": ["certificate", "database"],
    },
    "trust_rejected": {
        "failure": "secure connection rejected an outdated identity credential",
        "correct": "certificate",
        "distractors": ["memory", "config"],
    },
    "grammar_malformed": {
        "failure": "settings document cannot be parsed because its grammar is malformed",
        "correct": "config",
        "distractors": ["database", "cache"],
    },
    "transaction_owned": {
        "failure": "persistent store refuses writes because another transaction owns it",
        "correct": "database",
        "distractors": ["certificate", "memory"],
    },
    "allocation_failed": {
        "failure": "allocator cannot reserve additional bytes for compilation",
        "correct": "memory",
        "distractors": ["config", "cache"],
    },
}

NO_LEXICAL_OVERLAP_COMMANDS = {
    "cache": "purge obsolete cache files",
    "certificate": "rotate TLS certificate",
    "config": "repair configuration syntax",
    "database": "release database lock",
    "memory": "reduce memory use",
}


def _command(candidate: str, marker: str) -> dict[str, Any]:
    return {
        "action_type": "RUN_COMMAND",
        "argv": [
            "<PYTHON>",
            "-c",
            f"print('{NO_LEXICAL_OVERLAP_COMMANDS[candidate]}: {marker}')",
        ],
        "expected_exit_code": 0,
    }


def _lexical_overlap(failure: str, command: str) -> list[str]:
    return sorted(set(_semantic_words(failure)) & set(_semantic_words(command)))


def _manifest(
    *,
    workspace_root: str | Path,
    repository_id: str,
    erased: bool,
) -> dict[str, Any]:
    episodes = []
    for recipe_id, recipe in NO_LEXICAL_OVERLAP_RECIPES.items():
        marker = f"phase2ca_{repository_id}_candidate"
        correct = str(recipe["correct"])
        correct_command = NO_LEXICAL_OVERLAP_COMMANDS[correct]
        overlap = _lexical_overlap(str(recipe["failure"]), correct_command)
        if overlap:
            raise ValueError(f"phase2ca recipe {recipe_id!r} has lexical overlap: {overlap}")
        order = [*recipe["distractors"], correct]
        commands = {candidate: _command(candidate, marker) for candidate in order}
        episodes.append(
            {
                "episode_id": f"phase2ca_{repository_id}_{recipe_id}_{'erased' if erased else 'visible'}",
                "task_type": "test_failure_reflex",
                "description": "Recover from a paraphrased failure using an allowlisted candidate.",
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
                            "process exited; semantic details withheld"
                            if erased
                            else str(recipe["failure"])
                        ),
                        "stderr_unread": True,
                        "stderr_lines": 1,
                        "last_output_channel": "stderr",
                    },
                },
                "no_lexical_overlap_contract": {
                    "recipe_id": recipe_id,
                    "semantic_signal_erased": erased,
                    "candidate_order": order,
                    "correct_candidate": correct,
                    "correct_candidate_is_first": order[0] == correct,
                    "failure_correct_command_overlap": overlap,
                    "expected_sequence_exposed": False,
                },
            }
        )
    return {
        "workspace_root": str(Path(workspace_root).resolve()),
        "generated_by": {
            "phase": "phase2ca",
            "repository_id": repository_id,
            "semantic_signal_erased": erased,
        },
        "episodes": episodes,
    }


def _policy_factories(
    *,
    checkpoint_path: str | Path,
    cortex_matcher: CausalLMConditionalSemanticMatcher,
    lexical_matcher: HashedDualEncoderSemanticMatcher,
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
        "frozen_causal_cortex_native": lambda: EventGatedSequencePolicy(
            policy_label="phase2ca_frozen_causal_cortex_native",
            semantic_command_prior_weight=8.0,
            semantic_command_scorer=cortex_matcher,
            **common,
        ),
        "learned_lexical_residual_native": lambda: EventGatedSequencePolicy(
            policy_label="phase2ca_learned_lexical_residual_native",
            semantic_command_prior_weight=8.0,
            semantic_command_scorer=lexical_matcher,
            **common,
        ),
        "equivariant_no_semantic_prior": lambda: EventGatedSequencePolicy(
            policy_label="phase2ca_equivariant_no_semantic_prior",
            **common,
        ),
        "bounded_state_rule": lambda: BoundedStateRulePolicy(
            policy_label="phase2ca_bounded_state_rule"
        ),
    }


def _run_suite(
    *,
    rows: list[dict[str, str]],
    checkpoint_path: str | Path,
    cortex_matcher: CausalLMConditionalSemanticMatcher,
    lexical_matcher: HashedDualEncoderSemanticMatcher,
    output_root: Path,
    timeout_seconds: float,
) -> dict[str, dict[str, Any]]:
    reports = {}
    for policy_id, factory in _policy_factories(
        checkpoint_path=checkpoint_path,
        cortex_matcher=cortex_matcher,
        lexical_matcher=lexical_matcher,
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
                    policy_label=f"phase2ca_{policy_id}",
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


def run_phase2ca_no_lexical_overlap_transfer(
    *,
    checkpoint_path: str | Path,
    phase2bq_report_json: str | Path,
    lexical_matcher_path: str | Path,
    cortex_model_path: str | Path,
    cortex_device: str,
    cortex_dtype: str,
    output_dir: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    phase2bq_report = json.loads(Path(phase2bq_report_json).read_text(encoding="utf-8-sig"))
    output_root = Path(output_dir)
    cortex_matcher = CausalLMConditionalSemanticMatcher.from_pretrained(
        cortex_model_path,
        device=cortex_device,
        dtype=cortex_dtype,
    )
    lexical_matcher = HashedDualEncoderSemanticMatcher.load(lexical_matcher_path)
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
        cortex_matcher=cortex_matcher,
        lexical_matcher=lexical_matcher,
        output_root=output_root / "visible",
        timeout_seconds=timeout_seconds,
    )
    erased_reports = _run_suite(
        rows=suites["erased"],
        checkpoint_path=checkpoint_path,
        cortex_matcher=cortex_matcher,
        lexical_matcher=lexical_matcher,
        output_root=output_root / "erased",
        timeout_seconds=timeout_seconds,
    )
    cortex_visible = visible_reports["frozen_causal_cortex_native"]["metrics"]
    cortex_erased = erased_reports["frozen_causal_cortex_native"]["metrics"]
    lexical_visible = visible_reports["learned_lexical_residual_native"]["metrics"]
    no_prior_visible = visible_reports["equivariant_no_semantic_prior"]["metrics"]
    rule_visible = visible_reports["bounded_state_rule"]["metrics"]
    checks = {
        "phase2bq_source_report_passed": phase2bq_report.get("passed") is True,
        "all_failure_correct_command_pairs_have_zero_lexical_overlap": all(
            not _lexical_overlap(
                str(recipe["failure"]),
                NO_LEXICAL_OVERLAP_COMMANDS[str(recipe["correct"])],
            )
            for recipe in NO_LEXICAL_OVERLAP_RECIPES.values()
        ),
        "cortex_solves_no_lexical_overlap_visible_tasks": cortex_visible[
            "task_completion_success_rate"
        ]
        == 1.0,
        "semantic_erasure_reduces_cortex_transfer": cortex_erased[
            "task_completion_success_rate"
        ]
        < cortex_visible["task_completion_success_rate"],
        "cortex_outperforms_lexical_residual": cortex_visible[
            "task_completion_success_rate"
        ]
        > lexical_visible["task_completion_success_rate"],
        "cortex_outperforms_no_prior": cortex_visible["task_completion_success_rate"]
        > no_prior_visible["task_completion_success_rate"],
        "cortex_outperforms_state_rule": cortex_visible["task_completion_success_rate"]
        > rule_visible["task_completion_success_rate"],
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2ca_no_lexical_overlap_transfer",
        "passed": passed,
        "ready_for_bounded_no_lexical_overlap_semantic_transfer_claim": passed,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checkpoint_path": str(checkpoint_path),
        "phase2bq_report_json": str(phase2bq_report_json),
        "lexical_matcher_path": str(lexical_matcher_path),
        "cortex_matcher_metadata": cortex_matcher.metadata(),
        "checks": checks,
        "visible_policy_reports": visible_reports,
        "erased_policy_reports": erased_reports,
        "comparison": {
            "cortex_visible_completion": cortex_visible["task_completion_success_rate"],
            "cortex_erased_completion": cortex_erased["task_completion_success_rate"],
            "lexical_visible_completion": lexical_visible["task_completion_success_rate"],
            "no_prior_visible_completion": no_prior_visible["task_completion_success_rate"],
            "rule_visible_completion": rule_visible["task_completion_success_rate"],
        },
        "supported_claims": [
            "a frozen causal language cortex enabled bounded semantic action selection for paraphrased failures with zero normalized lexical overlap"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "unbounded semantic perception",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2cb_natural_failure_no_lexical_overlap_transfer"
            if passed
            else "repair_phase2ca_no_lexical_overlap_transfer"
        ),
    }
    output_path = Path(output_report_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate bounded semantic transfer with zero lexical overlap."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--phase2bq-report-json", required=True)
    parser.add_argument("--lexical-matcher-path", required=True)
    parser.add_argument("--cortex-model-path", required=True)
    parser.add_argument("--cortex-device", default="cpu")
    parser.add_argument("--cortex-dtype", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2ca_no_lexical_overlap_transfer(
        checkpoint_path=args.checkpoint_path,
        phase2bq_report_json=args.phase2bq_report_json,
        lexical_matcher_path=args.lexical_matcher_path,
        cortex_model_path=args.cortex_model_path,
        cortex_device=args.cortex_device,
        cortex_dtype=args.cortex_dtype,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
