from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from reflexlm.baselines.state_rule_policy import BoundedStateRulePolicy
from reflexlm.cli.run_phase2bn_model_selected_sealed_runtime import (
    run_phase2bn_model_selected_sealed_runtime,
)
from reflexlm.cli.run_phase2br_runtime_ablation_comparison import _aggregate_subreports
from reflexlm.eval import EventGatedSequencePolicy
from reflexlm.models.semantic_matcher import (
    FrozenEncoderDualSemanticMatcher,
    HashedDualEncoderSemanticMatcher,
    _semantic_words,
)
from reflexlm.train import load_model_checkpoint


NATURAL_INTENT_COMMANDS = {
    "replace_attribute": "restore missing method",
    "insert_import": "load required library",
    "replace_literal": "repair output mismatch",
}

_FAILURE_LINE_RE = re.compile(
    r"(?:E\s+)?((?:AttributeError|NameError|ImportError|ModuleNotFoundError|"
    r"AssertionError|SyntaxError|TypeError|ValueError|KeyError):[^\n]*)"
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _operation(row: dict[str, Any]) -> str:
    target = row.get("learned_patch_descriptor_target")
    return str(target.get("operation") or "") if isinstance(target, dict) else ""


def _failure_line(row: dict[str, Any]) -> str | None:
    evidence = row.get("runtime_visible_evidence")
    if not isinstance(evidence, dict):
        return None
    pytest = evidence.get("pytest_before_patch")
    if not isinstance(pytest, dict):
        return None
    match = _FAILURE_LINE_RE.search(str(pytest.get("stdout_excerpt") or ""))
    return match.group(1).strip() if match else None


def _changed_paths(row: dict[str, Any]) -> list[str]:
    evidence = row.get("runtime_visible_evidence")
    values = evidence.get("changed_files", []) if isinstance(evidence, dict) else []
    return [str(value) for value in values if str(value).strip()]


def _receptor_source_text(row: dict[str, Any]) -> str:
    return " ".join([str(_failure_line(row) or ""), *_changed_paths(row)]).strip()


def _lexical_overlap(row: dict[str, Any]) -> list[str]:
    operation = _operation(row)
    command = NATURAL_INTENT_COMMANDS.get(operation, "")
    return sorted(set(_semantic_words(_receptor_source_text(row))) & set(_semantic_words(command)))


def _eligible_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for row in _read_jsonl(path):
        contract = row.get("runtime_visible_contract")
        if row.get("source_kind") != "public_repo":
            continue
        if not isinstance(contract, dict) or contract.get("no_gold_hint") is not True:
            continue
        if _operation(row) not in NATURAL_INTENT_COMMANDS:
            continue
        if not _failure_line(row) or _lexical_overlap(row):
            continue
        rows.append(row)
    return rows


def _command(text: str, marker: str) -> dict[str, Any]:
    return {
        "action_type": "RUN_COMMAND",
        "argv": ["<PYTHON>", "-c", f"print('{text}: {marker}')"],
        "expected_exit_code": 0,
    }


def _candidate_order(operation: str, index: int) -> list[str]:
    others = [value for value in NATURAL_INTENT_COMMANDS if value != operation]
    return (
        [others[0], operation, others[1]]
        if index % 2 == 0
        else [others[0], others[1], operation]
    )


def _wrong_donor(rows: list[dict[str, Any]], index: int) -> dict[str, Any]:
    operation = _operation(rows[index])
    for offset in range(1, len(rows)):
        candidate = rows[(index + offset) % len(rows)]
        if _operation(candidate) != operation:
            return candidate
    raise ValueError("phase2cb wrong-receptor control requires multiple operations")


def _manifest(
    *,
    rows: list[dict[str, Any]],
    workspace_root: str | Path,
    suite_id: str,
) -> dict[str, Any]:
    if suite_id not in {"visible", "erased", "wrong_receptor"}:
        raise ValueError(f"unsupported phase2cb suite {suite_id!r}")
    episodes = []
    for index, row in enumerate(rows):
        operation = _operation(row)
        source = row if suite_id == "visible" else None
        donor = _wrong_donor(rows, index) if suite_id == "wrong_receptor" else None
        receptor_row = source or donor
        failure = (
            _failure_line(receptor_row)
            if receptor_row is not None
            else "public repository process exited; semantic failure details withheld"
        )
        changed_paths = _changed_paths(receptor_row) if receptor_row is not None else []
        order = _candidate_order(operation, index)
        marker = f"phase2cb_{index:05d}_candidate"
        commands = {
            candidate: _command(NATURAL_INTENT_COMMANDS[candidate], marker)
            for candidate in order
        }
        correct = commands[operation]
        episodes.append(
            {
                "episode_id": f"phase2cb_{suite_id}_{index:05d}",
                "task_type": "test_failure_reflex",
                "description": "Select a bounded recovery intent from natural public-repository failure receptors.",
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
                "initial_state": {
                    "process": {"status": "exited", "exit_code": 1},
                    "terminal": {
                        "stderr_delta": str(failure),
                        "stderr_unread": True,
                        "stderr_lines": 1,
                        "last_output_channel": "stderr",
                    },
                    "filesystem": {
                        "changed_paths": [],
                        "dirty_files": [],
                        "external_change_detected": False,
                    },
                    "runtime_evidence": {
                        "terminal_observations": changed_paths,
                    },
                },
                "natural_failure_contract": {
                    "source_task_id": row.get("task_id"),
                    "source_kind": row.get("source_kind"),
                    "repo_origin": row.get("repo_origin"),
                    "repo_commit": row.get("repo_commit"),
                    "operation": operation,
                    "suite_id": suite_id,
                    "wrong_receptor_source_task_id": donor.get("task_id") if donor else None,
                    "correct_candidate_is_first": order[0] == operation,
                    "failure_correct_command_overlap": _lexical_overlap(row),
                    "expected_sequence_exposed": False,
                    "free_form_action_generation": False,
                },
            }
        )
    return {
        "workspace_root": str(Path(workspace_root).resolve()),
        "generated_by": {"phase": "phase2cb", "suite_id": suite_id},
        "episodes": episodes,
    }


def _policy_factories(
    *,
    checkpoint_path: str | Path,
    natural_matcher: FrozenEncoderDualSemanticMatcher,
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
        "trained_natural_cortex_native": lambda: EventGatedSequencePolicy(
            policy_label="phase2cb_trained_natural_cortex_native",
            semantic_command_prior_weight=8.0,
            semantic_command_scorer=natural_matcher,
            **common,
        ),
        "learned_lexical_residual_native": lambda: EventGatedSequencePolicy(
            policy_label="phase2cb_learned_lexical_residual_native",
            semantic_command_prior_weight=8.0,
            semantic_command_scorer=lexical_matcher,
            **common,
        ),
        "equivariant_no_semantic_prior": lambda: EventGatedSequencePolicy(
            policy_label="phase2cb_equivariant_no_semantic_prior",
            **common,
        ),
        "bounded_state_rule": lambda: BoundedStateRulePolicy(
            policy_label="phase2cb_bounded_state_rule"
        ),
    }


def _run_suite(
    *,
    manifest_json: str | Path,
    checkpoint_path: str | Path,
    natural_matcher: FrozenEncoderDualSemanticMatcher,
    lexical_matcher: HashedDualEncoderSemanticMatcher,
    output_root: Path,
    timeout_seconds: float,
) -> dict[str, dict[str, Any]]:
    reports = {}
    for policy_id, factory in _policy_factories(
        checkpoint_path=checkpoint_path,
        natural_matcher=natural_matcher,
        lexical_matcher=lexical_matcher,
    ).items():
        policy = factory()
        policy_root = output_root / policy_id
        report = run_phase2bn_model_selected_sealed_runtime(
            checkpoint_path=None,
            manifest_json=manifest_json,
            output_jsonl=policy_root / "trajectories.jsonl",
            output_report_json=policy_root / "report.json",
            timeout_seconds=timeout_seconds,
            max_extra_steps=3,
            policy_label=f"phase2cb_{policy_id}",
            policy_instance=policy,
        )
        reports[policy_id] = {
            "metrics": _aggregate_subreports([report]),
            "report_json": str(policy_root / "report.json"),
        }
    return reports


def run_phase2cb_natural_failure_no_lexical_overlap_transfer(
    *,
    checkpoint_path: str | Path,
    train_jsonl: str | Path,
    holdout_jsonl: str | Path,
    lexical_matcher_path: str | Path,
    cortex_model_path: str | Path,
    cortex_device: str,
    cortex_dtype: str,
    output_dir: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    train_rows = _eligible_rows(train_jsonl)
    holdout_rows = _eligible_rows(holdout_jsonl)
    natural_matcher = FrozenEncoderDualSemanticMatcher.from_pretrained(
        cortex_model_path,
        device=cortex_device,
        dtype=cortex_dtype,
    )
    natural_matcher.fit(
        [
            (_receptor_source_text(row), NATURAL_INTENT_COMMANDS[_operation(row)])
            for row in train_rows
        ]
    )
    lexical_matcher = HashedDualEncoderSemanticMatcher.load(lexical_matcher_path)
    lexical_matcher.lexical_residual_weight = 3.0
    output_root = Path(output_dir)
    manifest_root = output_root / "manifests"
    manifests = {}
    for suite_id in ("visible", "erased", "wrong_receptor"):
        manifest_path = manifest_root / f"{suite_id}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                _manifest(rows=holdout_rows, workspace_root=Path.cwd(), suite_id=suite_id),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        manifests[suite_id] = manifest_path
    suite_reports = {
        suite_id: _run_suite(
            manifest_json=manifest_path,
            checkpoint_path=checkpoint_path,
            natural_matcher=natural_matcher,
            lexical_matcher=lexical_matcher,
            output_root=output_root / suite_id,
            timeout_seconds=timeout_seconds,
        )
        for suite_id, manifest_path in manifests.items()
    }
    natural_visible = suite_reports["visible"]["trained_natural_cortex_native"]["metrics"]
    natural_erased = suite_reports["erased"]["trained_natural_cortex_native"]["metrics"]
    natural_wrong = suite_reports["wrong_receptor"]["trained_natural_cortex_native"]["metrics"]
    lexical_visible = suite_reports["visible"]["learned_lexical_residual_native"]["metrics"]
    no_prior_visible = suite_reports["visible"]["equivariant_no_semantic_prior"]["metrics"]
    rule_visible = suite_reports["visible"]["bounded_state_rule"]["metrics"]
    train_task_ids = {str(row.get("task_id")) for row in train_rows}
    holdout_task_ids = {str(row.get("task_id")) for row in holdout_rows}
    checks = {
        "minimum_natural_train_rows_met": len(train_rows) >= 100,
        "minimum_natural_holdout_rows_met": len(holdout_rows) >= 100,
        "train_holdout_task_ids_disjoint": train_task_ids.isdisjoint(holdout_task_ids),
        "all_rows_are_public_repo_no_gold_hint": all(
            row.get("source_kind") == "public_repo"
            and row.get("runtime_visible_contract", {}).get("no_gold_hint") is True
            for row in [*train_rows, *holdout_rows]
        ),
        "all_holdout_pairs_have_zero_lexical_overlap": all(
            not _lexical_overlap(row) for row in holdout_rows
        ),
        "natural_cortex_holdout_completion_meets_gate": natural_visible[
            "task_completion_success_rate"
        ]
        >= 0.95,
        "erased_receptor_reduces_natural_transfer": natural_erased[
            "task_completion_success_rate"
        ]
        < natural_visible["task_completion_success_rate"],
        "wrong_receptor_reduces_natural_transfer": natural_wrong[
            "task_completion_success_rate"
        ]
        < natural_visible["task_completion_success_rate"],
        "natural_cortex_outperforms_lexical_residual": natural_visible[
            "task_completion_success_rate"
        ]
        > lexical_visible["task_completion_success_rate"],
        "natural_cortex_outperforms_no_prior": natural_visible[
            "task_completion_success_rate"
        ]
        > no_prior_visible["task_completion_success_rate"],
        "natural_cortex_outperforms_state_rule": natural_visible[
            "task_completion_success_rate"
        ]
        > rule_visible["task_completion_success_rate"],
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2cb_natural_failure_no_lexical_overlap_transfer",
        "passed": passed,
        "ready_for_bounded_natural_failure_semantic_transfer_claim": passed,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checkpoint_path": str(checkpoint_path),
        "train_jsonl": str(train_jsonl),
        "holdout_jsonl": str(holdout_jsonl),
        "lexical_matcher_path": str(lexical_matcher_path),
        "natural_matcher_metadata": natural_matcher.metadata(),
        "dataset": {
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "holdout_repo_origins": sorted(
                {str(row.get("repo_origin")) for row in holdout_rows}
            ),
            "operations": sorted(NATURAL_INTENT_COMMANDS),
        },
        "checks": checks,
        "suite_reports": suite_reports,
        "comparison": {
            "natural_visible_completion": natural_visible["task_completion_success_rate"],
            "natural_erased_completion": natural_erased["task_completion_success_rate"],
            "natural_wrong_receptor_completion": natural_wrong[
                "task_completion_success_rate"
            ],
            "lexical_visible_completion": lexical_visible["task_completion_success_rate"],
            "no_prior_visible_completion": no_prior_visible[
                "task_completion_success_rate"
            ],
            "rule_visible_completion": rule_visible["task_completion_success_rate"],
        },
        "supported_claims": [
            "a small learned action head over frozen cortex embeddings selected bounded recovery intents from natural public-repository failure receptors with zero normalized lexical overlap"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "arbitrary natural failure recovery",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2cc_continuous_natural_receptor_closed_loop"
            if passed
            else "repair_phase2cb_natural_failure_no_lexical_overlap_transfer"
        ),
    }
    output_path = Path(output_report_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate learned semantic action heads on natural public-repo failures."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--lexical-matcher-path", required=True)
    parser.add_argument("--cortex-model-path", required=True)
    parser.add_argument("--cortex-device", default="cpu")
    parser.add_argument("--cortex-dtype", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2cb_natural_failure_no_lexical_overlap_transfer(
        checkpoint_path=args.checkpoint_path,
        train_jsonl=args.train_jsonl,
        holdout_jsonl=args.holdout_jsonl,
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
