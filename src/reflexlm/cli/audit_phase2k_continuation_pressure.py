from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from reflexlm.cli.check_phase2d_gates import _metric, _trace_audit
from reflexlm.data.tasks import TaskType
from reflexlm.llm.candidate_features import source_overlap_command_slot_prediction
from reflexlm.llm.head_dataset import build_phase2c_head_state_prompt_from_state
from reflexlm.models.features import candidate_commands, serialize_state_as_text
from reflexlm.schema import ActionType, SystemStateFrame


REQUIRED_EVIDENCE_DENSITIES = {"low", "medium", "high"}
REQUIRED_CANDIDATE_COUNTS = {2, 3, 4}
REQUIRED_CONTINUATION_DEPTHS = {"one_step", "two_step", "stale_state_refresh"}
REQUIRED_AMBIGUITY_CLASSES = {"same_intent_command", "same_file_read", "stage_transition"}


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _group_by_episode(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("episode_id"))].append(row)
    return dict(grouped)


def _action_type(row: dict[str, Any]) -> str | None:
    action = row.get("action")
    if not isinstance(action, dict):
        return None
    return str(action.get("type"))


def _source_overlap_rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    correct = 0
    command_rows: list[dict[str, Any]] = []
    for row in rows:
        if _action_type(row) != ActionType.RUN_COMMAND.value:
            continue
        state = SystemStateFrame.model_validate(row["state"])
        candidates = candidate_commands(state)
        expected = row.get("action", {}).get("command")
        if expected not in candidates:
            continue
        total += 1
        prediction = source_overlap_command_slot_prediction(
            build_phase2c_head_state_prompt_from_state(state),
            candidates,
        )
        expected_slot = candidates.index(expected)
        correct += int(prediction == expected_slot)
        command_rows.append(
            {
                "episode_id": row.get("episode_id"),
                "candidate_count": len(candidates),
                "expected_slot": expected_slot,
                "source_overlap_prediction": prediction,
                "source_overlap_correct": prediction == expected_slot,
                "last_command_visible": bool(state.terminal.last_command),
            }
        )
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total if total else None),
        "examples": command_rows[:8],
    }


def _metadata_rollup(metadata_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episode_count": len(metadata_rows),
        "evidence_densities": sorted(
            {str(row.get("phase2k_evidence_density")) for row in metadata_rows if row.get("phase2k_evidence_density")}
        ),
        "candidate_counts": sorted(
            {
                int(row.get("phase2k_candidate_count"))
                for row in metadata_rows
                if row.get("phase2k_candidate_count") is not None
            }
        ),
        "continuation_depths": sorted(
            {
                str(row.get("phase2k_continuation_depth"))
                for row in metadata_rows
                if row.get("phase2k_continuation_depth")
            }
        ),
        "ambiguity_classes": sorted(
            {
                str(row.get("phase2k_ambiguity_class"))
                for row in metadata_rows
                if row.get("phase2k_ambiguity_class")
            }
        ),
    }


def build_phase2k_data_health(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    train_metadata_json: str | Path,
    val_metadata_json: str | Path,
    max_source_overlap_val_accuracy: float = 0.80,
) -> dict[str, Any]:
    train_rows = _load_jsonl(train_jsonl)
    val_rows = _load_jsonl(val_jsonl)
    train_metadata = _load_json(train_metadata_json)
    val_metadata = _load_json(val_metadata_json)
    train_meta_rows = train_metadata if isinstance(train_metadata, list) else []
    val_meta_rows = val_metadata if isinstance(val_metadata, list) else []
    train_rollup = _metadata_rollup(train_meta_rows)
    val_rollup = _metadata_rollup(val_meta_rows)
    source_overlap = {
        "train": _source_overlap_rollup(train_rows),
        "val": _source_overlap_rollup(val_rows),
    }
    serialized = "\n".join(
        serialize_state_as_text(SystemStateFrame.model_validate(row["state"]))
        for row in train_rows + val_rows
        if isinstance(row.get("state"), dict)
    )
    val_accuracy = source_overlap["val"]["accuracy"]
    command_state_without_last_command = all(
        not example["last_command_visible"] for example in source_overlap["val"]["examples"]
    )
    checks = {
        "phase2k_train_rows_present": len(train_rows) > 0,
        "phase2k_val_rows_present": len(val_rows) > 0,
        "phase2k_train_metadata_present": len(train_meta_rows) > 0,
        "phase2k_val_metadata_present": len(val_meta_rows) > 0,
        "phase2k_profiles_nonsealed": "external_trace_v3_semantic_required" not in serialized,
        "phase2k_no_hidden_hint_or_gold_label_visible": (
            "recovery_hint=" not in serialized
            and "correct_command" not in serialized
            and "scenario_template" not in serialized
        ),
        "phase2k_evidence_density_coverage": REQUIRED_EVIDENCE_DENSITIES.issubset(
            set(val_rollup["evidence_densities"])
        ),
        "phase2k_candidate_count_coverage": REQUIRED_CANDIDATE_COUNTS.issubset(
            set(val_rollup["candidate_counts"])
        ),
        "phase2k_continuation_depth_coverage": REQUIRED_CONTINUATION_DEPTHS.issubset(
            set(val_rollup["continuation_depths"])
        ),
        "phase2k_ambiguity_class_coverage": REQUIRED_AMBIGUITY_CLASSES.issubset(
            set(val_rollup["ambiguity_classes"])
        ),
        "phase2k_source_overlap_baseline_recorded": source_overlap["val"]["total"] > 0,
        "phase2k_source_overlap_baseline_below_threshold": (
            isinstance(val_accuracy, float) and val_accuracy <= max_source_overlap_val_accuracy
        ),
        "phase2k_prior_command_memory_required": command_state_without_last_command,
    }
    blocked_actions: list[str] = []
    if not all(checks.values()):
        blocked_actions.append("do_not_train_phase2k_until_data_health_passes")
    if not checks["phase2k_source_overlap_baseline_below_threshold"]:
        blocked_actions.append("do_not_train_when_source_overlap_solves_phase2k_val")
    if not checks["phase2k_prior_command_memory_required"]:
        blocked_actions.append("do_not_train_without_prior_command_memory_pressure")
    return {
        "audit_family": "phase2k_continuation_pressure_data_health",
        "passed": all(checks.values()),
        "allowed_next_action": (
            "run_phase2k_smoke_training_only"
            if all(checks.values())
            else "revise_phase2k_data_before_training"
        ),
        "blocked_actions": blocked_actions,
        "checks": checks,
        "thresholds": {
            "max_source_overlap_val_accuracy": max_source_overlap_val_accuracy,
            "required_evidence_densities": sorted(REQUIRED_EVIDENCE_DENSITIES),
            "required_candidate_counts": sorted(REQUIRED_CANDIDATE_COUNTS),
            "required_continuation_depths": sorted(REQUIRED_CONTINUATION_DEPTHS),
            "required_ambiguity_classes": sorted(REQUIRED_AMBIGUITY_CLASSES),
        },
        "rollups": {
            "train": train_rollup,
            "val": val_rollup,
            "source_overlap": source_overlap,
        },
        "inputs": {
            "train_jsonl": str(Path(train_jsonl)),
            "val_jsonl": str(Path(val_jsonl)),
            "train_metadata_json": str(Path(train_metadata_json)),
            "val_metadata_json": str(Path(val_metadata_json)),
        },
    }


def build_phase2k_postflight(
    *,
    data_health_json: str | Path,
    full_eval_json: str | Path,
    native_head_only_eval_json: str | Path,
    no_nsi_eval_json: str | Path | None = None,
    continuation_only_eval_json: str | Path | None = None,
    min_full_completion: float = 0.85,
    min_full_minus_native_head_only: float = 0.10,
    postflight_stage: str = "full",
) -> dict[str, Any]:
    if postflight_stage not in {"smoke", "full"}:
        raise ValueError("postflight_stage must be 'smoke' or 'full'")
    data_health = _load_json(data_health_json)
    full = _load_json(full_eval_json)
    native = _load_json(native_head_only_eval_json)
    no_nsi = _load_json(no_nsi_eval_json)
    continuation = _load_json(continuation_only_eval_json)
    full_completion = _metric(full, "task_completion_rate")
    native_completion = _metric(native, "task_completion_rate")
    no_nsi_completion = _metric(no_nsi, "task_completion_rate") if no_nsi else None
    continuation_completion = _metric(continuation, "task_completion_rate") if continuation else None
    full_minus_native = (
        full_completion - native_completion
        if isinstance(full_completion, float) and isinstance(native_completion, float)
        else None
    )
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "full_completion_gate_passed": (
            isinstance(full_completion, float) and full_completion >= min_full_completion
        ),
        "native_head_only_baseline_measured": isinstance(native_completion, float),
        "full_beats_native_head_only_by_required_delta": (
            isinstance(full_minus_native, float)
            and full_minus_native >= min_full_minus_native_head_only
        ),
        "source_overlap_baseline_below_threshold": data_health.get("checks", {}).get(
            "phase2k_source_overlap_baseline_below_threshold"
        )
        is True,
        "sealed_v3_not_used_for_postflight": "external_trace_v3_semantic_required"
        not in json.dumps(data_health.get("inputs", {}), sort_keys=True),
    }
    blocked_actions: list[str] = []
    if not checks["full_beats_native_head_only_by_required_delta"]:
        blocked_actions.append("do_not_package_without_full_beating_native_head_only")
    if not checks["full_completion_gate_passed"]:
        blocked_actions.append("do_not_package_until_phase2k_val_gate_passes")
    if not checks["data_health_passed"]:
        blocked_actions.append("do_not_package_until_phase2k_data_health_passes")
    passed = all(checks.values())
    ready_for_full_train = passed and postflight_stage == "smoke"
    ready_for_package = passed and postflight_stage == "full"
    allowed_next_action = "revise_phase2k_before_full_train"
    if postflight_stage == "smoke" and passed:
        allowed_next_action = "run_phase2k_full_nonsealed_training_only"
    elif postflight_stage == "full" and passed:
        allowed_next_action = "run_phase2k_package_only"
    elif postflight_stage == "full":
        allowed_next_action = "revise_phase2k_before_package"
    return {
        "audit_family": "phase2k_continuation_pressure_postflight",
        "postflight_stage": postflight_stage,
        "passed": passed,
        "ready_for_full_train": ready_for_full_train,
        "ready_for_package": ready_for_package,
        "ready_for_sealed_eval": False,
        "allowed_next_action": allowed_next_action,
        "blocked_actions": blocked_actions,
        "checks": checks,
        "metrics": {
            "full_completion": full_completion,
            "native_head_only_completion": native_completion,
            "no_nsi_completion": no_nsi_completion,
            "continuation_only_completion": continuation_completion,
            "full_minus_native_head_only": full_minus_native,
            "full_trace_audit": _trace_audit(full),
        },
        "thresholds": {
            "min_full_completion": min_full_completion,
            "min_full_minus_native_head_only": min_full_minus_native_head_only,
        },
        "inputs": {
            "data_health_json": str(Path(data_health_json)),
            "full_eval_json": str(Path(full_eval_json)),
            "native_head_only_eval_json": str(Path(native_head_only_eval_json)),
            "no_nsi_eval_json": str(Path(no_nsi_eval_json)) if no_nsi_eval_json else None,
            "continuation_only_eval_json": (
                str(Path(continuation_only_eval_json)) if continuation_only_eval_json else None
            ),
        },
    }


def build_phase2k_sealed_gate(
    *,
    full_eval_json: str | Path,
    no_nsi_eval_json: str | Path,
    native_head_only_eval_json: str | Path,
    continuation_only_eval_json: str | Path,
    prompt_only_eval_json: str | Path,
    react_eval_json: str | Path,
    min_full_minus_no_nsi: float = 0.15,
    min_full_minus_native_head_only: float = 0.10,
    min_full_minus_continuation_only: float = 0.15,
) -> dict[str, Any]:
    evals = {
        "full": _load_json(full_eval_json),
        "no_nsi": _load_json(no_nsi_eval_json),
        "native_head_only": _load_json(native_head_only_eval_json),
        "continuation_only": _load_json(continuation_only_eval_json),
        "prompt_only": _load_json(prompt_only_eval_json),
        "react": _load_json(react_eval_json),
    }
    table: dict[str, dict[str, float | str | None]] = {}
    for name, payload in evals.items():
        table[name] = {
            "policy_label": payload.get("policy", {}).get("policy_label"),
            "task_completion_rate": _metric(payload, "task_completion_rate"),
            "command_decision_accuracy": _metric(payload, "command_decision_accuracy"),
            "model_calls": _metric(payload, "model_calls"),
            "state_hallucination_rate": _metric(payload, "state_hallucination_rate"),
            "dataset_path": payload.get("dataset", {}).get("dataset_path"),
        }

    full_completion = table["full"]["task_completion_rate"]
    no_nsi_completion = table["no_nsi"]["task_completion_rate"]
    native_completion = table["native_head_only"]["task_completion_rate"]
    continuation_completion = table["continuation_only"]["task_completion_rate"]

    def _delta(other: float | str | None) -> float | None:
        if isinstance(full_completion, float) and isinstance(other, float):
            return full_completion - other
        return None

    deltas = {
        "full_minus_no_nsi": _delta(no_nsi_completion),
        "full_minus_native_head_only": _delta(native_completion),
        "full_minus_continuation_only": _delta(continuation_completion),
    }
    all_hallucination_zero = all(
        row["state_hallucination_rate"] == 0.0 for row in table.values()
    )
    full_low_level_qwen_calls_zero = table["full"]["model_calls"] == 0.0
    sealed_inputs = json.dumps(
        {name: row["dataset_path"] for name, row in table.items()}, sort_keys=True
    )
    checks = {
        "sealed_v3_inputs_only": "phase2i_external_trace_v3_semantic_required"
        in sealed_inputs,
        "full_beats_no_nsi_by_required_delta": (
            isinstance(deltas["full_minus_no_nsi"], float)
            and deltas["full_minus_no_nsi"] >= min_full_minus_no_nsi
        ),
        "full_beats_native_head_only_by_required_delta": (
            isinstance(deltas["full_minus_native_head_only"], float)
            and deltas["full_minus_native_head_only"]
            >= min_full_minus_native_head_only
        ),
        "full_beats_continuation_only_by_required_delta": (
            isinstance(deltas["full_minus_continuation_only"], float)
            and deltas["full_minus_continuation_only"]
            >= min_full_minus_continuation_only
        ),
        "allowlist_hallucination_zero": all_hallucination_zero,
        "full_low_level_qwen_calls_zero": full_low_level_qwen_calls_zero,
    }
    passed = all(checks.values())
    return {
        "audit_family": "phase2k_sealed_v3_gate",
        "passed": passed,
        "claim_boundary": (
            "sealed_v3_supports_full_package_necessity"
            if passed
            else "bounded_claim_only_do_not_upgrade_to_full_package_necessity"
        ),
        "checks": checks,
        "metrics": {
            "table": table,
            "deltas": deltas,
        },
        "thresholds": {
            "min_full_minus_no_nsi": min_full_minus_no_nsi,
            "min_full_minus_native_head_only": min_full_minus_native_head_only,
            "min_full_minus_continuation_only": min_full_minus_continuation_only,
        },
        "inputs": {
            "full_eval_json": str(Path(full_eval_json)),
            "no_nsi_eval_json": str(Path(no_nsi_eval_json)),
            "native_head_only_eval_json": str(Path(native_head_only_eval_json)),
            "continuation_only_eval_json": str(Path(continuation_only_eval_json)),
            "prompt_only_eval_json": str(Path(prompt_only_eval_json)),
            "react_eval_json": str(Path(react_eval_json)),
        },
    }


def render_phase2k_sealed_gate_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase2K Sealed v3 Gate",
        "",
        f"- Passed: `{str(report['passed']).lower()}`",
        f"- Claim boundary: `{report['claim_boundary']}`",
        "",
        "| mechanism | completion | command_accuracy | model_calls | hallucination |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["metrics"]["table"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(row["task_completion_rate"]),
                    str(row["command_decision_accuracy"]),
                    str(row["model_calls"]),
                    str(row["state_hallucination_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "| delta | value | threshold | passed |",
            "| --- | ---: | ---: | --- |",
            (
                f"| full_minus_no_nsi | {report['metrics']['deltas']['full_minus_no_nsi']} | "
                f"{report['thresholds']['min_full_minus_no_nsi']} | "
                f"{report['checks']['full_beats_no_nsi_by_required_delta']} |"
            ),
            (
                f"| full_minus_native_head_only | {report['metrics']['deltas']['full_minus_native_head_only']} | "
                f"{report['thresholds']['min_full_minus_native_head_only']} | "
                f"{report['checks']['full_beats_native_head_only_by_required_delta']} |"
            ),
            (
                f"| full_minus_continuation_only | {report['metrics']['deltas']['full_minus_continuation_only']} | "
                f"{report['thresholds']['min_full_minus_continuation_only']} | "
                f"{report['checks']['full_beats_continuation_only_by_required_delta']} |"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2K continuation-pressure gates.")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data-health")
    data.add_argument("--train-jsonl", required=True)
    data.add_argument("--val-jsonl", required=True)
    data.add_argument("--train-metadata-json", required=True)
    data.add_argument("--val-metadata-json", required=True)
    data.add_argument("--output-json")
    data.add_argument("--max-source-overlap-val-accuracy", type=float, default=0.80)
    data.add_argument("--no-fail", action="store_true")
    post = sub.add_parser("postflight")
    post.add_argument("--data-health-json", required=True)
    post.add_argument("--full-eval-json", required=True)
    post.add_argument("--native-head-only-eval-json", required=True)
    post.add_argument("--no-nsi-eval-json")
    post.add_argument("--continuation-only-eval-json")
    post.add_argument("--output-json")
    post.add_argument("--min-full-completion", type=float, default=0.85)
    post.add_argument("--min-full-minus-native-head-only", type=float, default=0.10)
    post.add_argument("--stage", choices=("smoke", "full"), default="full")
    post.add_argument("--no-fail", action="store_true")
    sealed = sub.add_parser("sealed-gate")
    sealed.add_argument("--full-eval-json", required=True)
    sealed.add_argument("--no-nsi-eval-json", required=True)
    sealed.add_argument("--native-head-only-eval-json", required=True)
    sealed.add_argument("--continuation-only-eval-json", required=True)
    sealed.add_argument("--prompt-only-eval-json", required=True)
    sealed.add_argument("--react-eval-json", required=True)
    sealed.add_argument("--output-json")
    sealed.add_argument("--output-md")
    sealed.add_argument("--min-full-minus-no-nsi", type=float, default=0.15)
    sealed.add_argument("--min-full-minus-native-head-only", type=float, default=0.10)
    sealed.add_argument("--min-full-minus-continuation-only", type=float, default=0.15)
    sealed.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    if args.command == "data-health":
        report = build_phase2k_data_health(
            train_jsonl=args.train_jsonl,
            val_jsonl=args.val_jsonl,
            train_metadata_json=args.train_metadata_json,
            val_metadata_json=args.val_metadata_json,
            max_source_overlap_val_accuracy=args.max_source_overlap_val_accuracy,
        )
    elif args.command == "postflight":
        report = build_phase2k_postflight(
            data_health_json=args.data_health_json,
            full_eval_json=args.full_eval_json,
            native_head_only_eval_json=args.native_head_only_eval_json,
            no_nsi_eval_json=args.no_nsi_eval_json,
            continuation_only_eval_json=args.continuation_only_eval_json,
            min_full_completion=args.min_full_completion,
            min_full_minus_native_head_only=args.min_full_minus_native_head_only,
            postflight_stage=args.stage,
        )
    else:
        report = build_phase2k_sealed_gate(
            full_eval_json=args.full_eval_json,
            no_nsi_eval_json=args.no_nsi_eval_json,
            native_head_only_eval_json=args.native_head_only_eval_json,
            continuation_only_eval_json=args.continuation_only_eval_json,
            prompt_only_eval_json=args.prompt_only_eval_json,
            react_eval_json=args.react_eval_json,
            min_full_minus_no_nsi=args.min_full_minus_no_nsi,
            min_full_minus_native_head_only=args.min_full_minus_native_head_only,
            min_full_minus_continuation_only=args.min_full_minus_continuation_only,
        )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if getattr(args, "output_md", None):
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_phase2k_sealed_gate_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
