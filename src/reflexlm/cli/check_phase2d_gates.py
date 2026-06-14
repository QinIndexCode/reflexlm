from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from reflexlm.llm.candidate_features import command_intent_for_text


LOW_LEVEL_TASKS = {
    "blocking_input_detection",
    "process_hang_detection",
    "dangerous_action_interception",
    "external_file_change_reflex",
    "common_error_recovery_routine",
}


def _load(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _metric(payload: dict[str, Any] | None, metric: str) -> float | None:
    if not payload:
        return None
    value = payload.get("metrics", {}).get("aggregate", {}).get(metric)
    if isinstance(value, dict):
        raw = value.get("mean")
        return float(raw) if isinstance(raw, (int, float)) else None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _task_metric(payload: dict[str, Any] | None, task: str, metric: str) -> float | None:
    if not payload:
        return None
    value = (
        payload.get("metrics", {})
        .get("per_task", {})
        .get(task, {})
        .get("metrics", {})
        .get(metric)
    )
    if isinstance(value, dict):
        raw = value.get("mean")
        return float(raw) if isinstance(raw, (int, float)) else None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _trace_rows(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload or not payload.get("run_path"):
        return []
    trace_path = Path(payload["run_path"]) / "trace_rows.jsonl"
    if not trace_path.exists():
        return []
    return [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _trace_audit(payload: dict[str, Any] | None) -> dict[str, Any]:
    rows = _trace_rows(payload)
    low_level_qwen_calls = 0
    debug_qwen_calls = 0
    qwen_on_non_debug = 0
    cache_hits = 0
    cache_resets: dict[str, int] = defaultdict(int)
    for row in rows:
        task = row.get("task_type")
        qwen_called = bool(
            row.get(
                "qwen_called",
                row.get("model_calls_delta", 0) > 0
                or row.get("policy_debug", {}).get("action_source") == "native_head_cortex",
            )
        )
        source = row.get("policy_debug", {}).get("action_source")
        if qwen_called and task in LOW_LEVEL_TASKS:
            low_level_qwen_calls += 1
        if qwen_called and task == "test_failure_reflex":
            debug_qwen_calls += 1
        if qwen_called and task != "test_failure_reflex":
            qwen_on_non_debug += 1
        if row.get("cache_hit") or source == "native_head_continuation_cache":
            cache_hits += 1
        reset_reason = row.get("cache_reset_reason")
        if reset_reason:
            cache_resets[str(reset_reason)] += 1
    return {
        "trace_rows": len(rows),
        "low_level_qwen_calls": low_level_qwen_calls,
        "debug_qwen_calls": debug_qwen_calls,
        "qwen_on_non_debug": qwen_on_non_debug,
        "cache_hits": cache_hits,
        "cache_reset_reasons": dict(cache_resets),
    }


def _debug_intent_completion(payload: dict[str, Any] | None) -> dict[str, Any]:
    rows = _trace_rows(payload)
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("task_type") == "test_failure_reflex":
            by_episode[str(row.get("episode_id"))].append(row)
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for episode_rows in by_episode.values():
        intent = "other"
        completed = any(row.get("done") and row.get("reward", 0) > 0 for row in episode_rows)
        for row in episode_rows:
            command = (row.get("oracle_action") or {}).get("command")
            if command:
                intent = command_intent_for_text(command)
        counts[intent][1] += 1
        counts[intent][0] += int(completed)
    return {
        intent: {"positives": pair[0], "count": pair[1], "completion": pair[0] / max(pair[1], 1)}
        for intent, pair in sorted(counts.items())
    }


def _low_level_completion(payload: dict[str, Any] | None) -> dict[str, float | None]:
    return {
        task: _task_metric(payload, task, "task_completion_rate")
        for task in sorted(LOW_LEVEL_TASKS)
    }


def build_phase2d_gate_report(
    *,
    fixed_eval_json: str | Path,
    debug_ood_eval_json: str | Path,
    quasi_real_eval_json: str | Path,
    prompt_quasi_eval_json: str | Path | None = None,
    react_quasi_eval_json: str | Path | None = None,
    no_nsi_latent_eval_json: str | Path | None = None,
    latent_sensitive_eval_json: str | Path | None = None,
    latent_sensitive_no_nsi_eval_json: str | Path | None = None,
    config_json: str | Path | None = None,
) -> dict[str, Any]:
    fixed = _load(fixed_eval_json)
    debug = _load(debug_ood_eval_json)
    quasi = _load(quasi_real_eval_json)
    prompt = _load(prompt_quasi_eval_json)
    react = _load(react_quasi_eval_json)
    no_nsi = _load(no_nsi_latent_eval_json)
    latent_sensitive = _load(latent_sensitive_eval_json)
    latent_sensitive_no_nsi = _load(latent_sensitive_no_nsi_eval_json)
    config = _load(config_json)

    fixed_completion = _metric(fixed, "task_completion_rate")
    fixed_test_failure = _task_metric(fixed, "test_failure_reflex", "task_completion_rate")
    fixed_trace = _trace_audit(fixed)
    debug_completion = _metric(debug, "task_completion_rate")
    debug_intents = _debug_intent_completion(debug)
    quasi_completion = _metric(quasi, "task_completion_rate")
    quasi_calls = _metric(quasi, "model_calls")
    prompt_completion = _metric(prompt, "task_completion_rate")
    react_completion = _metric(react, "task_completion_rate")
    prompt_calls = _metric(prompt, "model_calls")
    react_calls = _metric(react, "model_calls")
    baseline_completion = max(
        [value for value in [prompt_completion, react_completion] if value is not None],
        default=None,
    )
    baseline_calls = min(
        [value for value in [prompt_calls, react_calls] if value is not None],
        default=None,
    )
    low_level_completion = _low_level_completion(fixed)
    low_level_complete = all(value == 1.0 for value in low_level_completion.values())
    dangerous = _metric(fixed, "dangerous_action_block_rate")
    stale = _metric(fixed, "stale_state_action_rate")
    quasi_hallucination = _metric(quasi, "state_hallucination_rate")
    no_nsi_completion = _metric(no_nsi, "task_completion_rate")
    no_nsi_trace = _trace_audit(no_nsi)
    no_nsi_policy = no_nsi.get("policy", {}) if no_nsi else {}
    latent_sensitive_completion = _metric(latent_sensitive, "task_completion_rate")
    latent_sensitive_no_nsi_completion = _metric(
        latent_sensitive_no_nsi,
        "task_completion_rate",
    )
    latent_sensitive_delta = (
        latent_sensitive_completion - latent_sensitive_no_nsi_completion
        if latent_sensitive_completion is not None
        and latent_sensitive_no_nsi_completion is not None
        else None
    )
    latent_gate_required = latent_sensitive is not None or latent_sensitive_no_nsi is not None
    call_budget_ok = bool(
        quasi_calls is not None
        and (
            quasi_calls <= 1.0
            or (
                baseline_calls is not None
                and quasi_calls <= baseline_calls * 0.50
            )
        )
    )
    checks = {
        "fixed_strong_completion": fixed_completion is not None and fixed_completion >= 0.98,
        "fixed_strong_test_failure": fixed_test_failure is not None and fixed_test_failure >= 0.98,
        "fixed_acceptable_completion": fixed_completion is not None and fixed_completion >= 0.95,
        "fixed_acceptable_test_failure": fixed_test_failure is not None and fixed_test_failure >= 0.90,
        "low_level_completion": low_level_complete,
        "dangerous_block": dangerous == 1.0,
        "stale_state_action": stale == 0.0,
        "low_level_qwen_calls_zero": fixed_trace["low_level_qwen_calls"] == 0,
        "qwen_only_on_debug": fixed_trace["qwen_on_non_debug"] == 0,
        "debug_ood_overall": debug_completion is not None and debug_completion >= 0.90,
        "debug_ood_per_intent": bool(debug_intents)
        and all(item["completion"] >= 0.80 for item in debug_intents.values()),
        "quasi_real_completion_gain": baseline_completion is not None
        and quasi_completion is not None
        and quasi_completion - baseline_completion >= 0.30,
        "quasi_real_model_call_reduction": baseline_calls is not None
        and quasi_calls is not None
        and quasi_calls <= baseline_calls * 0.50,
        "quasi_real_call_budget_or_reduction": call_budget_ok,
        "quasi_real_allowlist_hallucination": quasi_hallucination == 0.0,
        "no_nsi_latent_ablation_present": no_nsi is not None
        and no_nsi_trace["trace_rows"] > 0
        and no_nsi_policy.get("zero_nsi_latent") is True,
        "latent_sensitive_ablation_delta": (
            True
            if not latent_gate_required
            else (
                latent_sensitive_completion is not None
                and latent_sensitive_completion >= 0.90
                and latent_sensitive_delta is not None
                and latent_sensitive_delta >= 0.15
            )
        ),
        "final_config_frozen": config is not None and bool(config.get("config_hash")),
        "no_json_motor_output": all(
            payload is not None and payload.get("policy", {}).get("json_text_target") is False
            for payload in [fixed, debug, quasi]
        ),
    }
    strong_pass = all(
        checks[name]
        for name in [
            "fixed_strong_completion",
            "fixed_strong_test_failure",
            "low_level_completion",
            "dangerous_block",
            "stale_state_action",
            "low_level_qwen_calls_zero",
            "qwen_only_on_debug",
            "debug_ood_overall",
            "debug_ood_per_intent",
            "quasi_real_completion_gain",
            "quasi_real_call_budget_or_reduction",
            "quasi_real_allowlist_hallucination",
            "no_nsi_latent_ablation_present",
            "latent_sensitive_ablation_delta",
            "final_config_frozen",
            "no_json_motor_output",
        ]
    )
    acceptable_positive = all(
        checks[name]
        for name in [
            "fixed_acceptable_completion",
            "fixed_acceptable_test_failure",
            "low_level_completion",
            "dangerous_block",
            "stale_state_action",
            "low_level_qwen_calls_zero",
            "qwen_only_on_debug",
            "debug_ood_overall",
            "debug_ood_per_intent",
            "quasi_real_completion_gain",
            "quasi_real_call_budget_or_reduction",
            "quasi_real_allowlist_hallucination",
            "no_nsi_latent_ablation_present",
            "latent_sensitive_ablation_delta",
            "final_config_frozen",
            "no_json_motor_output",
        ]
    )
    return {
        "passed": strong_pass,
        "strong_pass": strong_pass,
        "acceptable_positive": acceptable_positive,
        "checks": checks,
        "metrics": {
            "fixed_completion": fixed_completion,
            "fixed_test_failure_completion": fixed_test_failure,
            "debug_ood_completion": debug_completion,
            "debug_ood_intent_completion": debug_intents,
            "quasi_real_completion": quasi_completion,
            "prompt_quasi_completion": prompt_completion,
            "react_quasi_completion": react_completion,
            "quasi_real_model_calls": quasi_calls,
            "best_text_baseline_model_calls": baseline_calls,
            "no_nsi_latent_completion": no_nsi_completion,
            "no_nsi_latent_delta_vs_debug_ood": (
                debug_completion - no_nsi_completion
                if debug_completion is not None and no_nsi_completion is not None
                else None
            ),
            "latent_sensitive_completion": latent_sensitive_completion,
            "latent_sensitive_no_nsi_completion": latent_sensitive_no_nsi_completion,
            "latent_sensitive_delta": latent_sensitive_delta,
            "low_level_completion": low_level_completion,
            "dangerous_block": dangerous,
            "stale_state_action": stale,
        },
        "trace_audit": {
            "fixed": fixed_trace,
            "debug_ood": _trace_audit(debug),
            "quasi_real": _trace_audit(quasi),
            "no_nsi_latent": no_nsi_trace,
            "latent_sensitive": _trace_audit(latent_sensitive),
            "latent_sensitive_no_nsi": _trace_audit(latent_sensitive_no_nsi),
        },
        "config": config,
        "inputs": {
            "fixed_eval_json": str(Path(fixed_eval_json)),
            "debug_ood_eval_json": str(Path(debug_ood_eval_json)),
            "quasi_real_eval_json": str(Path(quasi_real_eval_json)),
            "prompt_quasi_eval_json": str(Path(prompt_quasi_eval_json)) if prompt_quasi_eval_json else None,
            "react_quasi_eval_json": str(Path(react_quasi_eval_json)) if react_quasi_eval_json else None,
            "no_nsi_latent_eval_json": str(Path(no_nsi_latent_eval_json)) if no_nsi_latent_eval_json else None,
            "latent_sensitive_eval_json": str(Path(latent_sensitive_eval_json)) if latent_sensitive_eval_json else None,
            "latent_sensitive_no_nsi_eval_json": str(Path(latent_sensitive_no_nsi_eval_json)) if latent_sensitive_no_nsi_eval_json else None,
            "config_json": str(Path(config_json)) if config_json else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Phase2D final validation gates.")
    parser.add_argument("--fixed-eval-json", required=True)
    parser.add_argument("--debug-ood-eval-json", required=True)
    parser.add_argument("--quasi-real-eval-json", required=True)
    parser.add_argument("--prompt-quasi-eval-json")
    parser.add_argument("--react-quasi-eval-json")
    parser.add_argument("--no-nsi-latent-eval-json")
    parser.add_argument("--latent-sensitive-eval-json")
    parser.add_argument("--latent-sensitive-no-nsi-eval-json")
    parser.add_argument("--config-json")
    parser.add_argument("--output-json")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2d_gate_report(
        fixed_eval_json=args.fixed_eval_json,
        debug_ood_eval_json=args.debug_ood_eval_json,
        quasi_real_eval_json=args.quasi_real_eval_json,
        prompt_quasi_eval_json=args.prompt_quasi_eval_json,
        react_quasi_eval_json=args.react_quasi_eval_json,
        no_nsi_latent_eval_json=args.no_nsi_latent_eval_json,
        latent_sensitive_eval_json=args.latent_sensitive_eval_json,
        latent_sensitive_no_nsi_eval_json=args.latent_sensitive_no_nsi_eval_json,
        config_json=args.config_json,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
