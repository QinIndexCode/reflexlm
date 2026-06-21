from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ReflexCoreRuntimeEvidenceAuditConfig:
    matrix_report_json: Path
    output_json: Path | None = None
    min_profile_runs: int = 9
    min_profile_pass_rate: float = 1.0
    min_live_episode_count: int = 15
    min_runtime_observation_steps: int = 30
    min_changed_file_observation_steps: int = 10
    min_terminal_observation_steps: int = 10
    min_observed_prediction_error_examples: int = 30
    require_all_profile_runs_passed: bool = True
    require_live_observation: bool = True


def audit_reflexcore_runtime_evidence(
    config: ReflexCoreRuntimeEvidenceAuditConfig,
) -> dict[str, object]:
    payload = _read_json(config.matrix_report_json)
    profile_runs = [_object(row) for row in _list(payload.get("profile_runs"))]
    checks = {
        "matrix_report_passed": _bool_check(
            payload.get("passed") is True,
            required=True,
            source="matrix_report",
        ),
        "profile_run_count": _min_check(
            len(profile_runs),
            config.min_profile_runs,
            source="profile_runs",
        ),
        "profile_pass_rate": _min_check(
            _number(payload.get("profile_pass_rate")),
            config.min_profile_pass_rate,
            source="matrix_report",
        ),
        "all_profile_runs_passed": _all_rows_check(
            profile_runs,
            field="passed",
            required=True,
            enabled=config.require_all_profile_runs_passed,
            source="profile_runs",
        ),
        "live_observation_enabled": _all_rows_check(
            profile_runs,
            field="real_sandbox_live_observation",
            required=True,
            enabled=config.require_live_observation,
            source="profile_runs",
        ),
        "live_episode_count_floor": _row_min_check(
            profile_runs,
            field="real_sandbox_live_episode_count",
            required_min=config.min_live_episode_count,
            source="profile_runs",
        ),
        "runtime_observation_steps_floor": _row_min_check(
            profile_runs,
            field="real_sandbox_runtime_observation_steps",
            required_min=config.min_runtime_observation_steps,
            source="profile_runs",
        ),
        "changed_file_observation_steps_floor": _row_min_check(
            profile_runs,
            field="real_sandbox_changed_file_observation_steps",
            required_min=config.min_changed_file_observation_steps,
            source="profile_runs",
        ),
        "terminal_observation_steps_floor": _row_min_check(
            profile_runs,
            field="real_sandbox_terminal_observation_steps",
            required_min=config.min_terminal_observation_steps,
            source="profile_runs",
        ),
        "observed_prediction_error_examples_floor": _row_min_check(
            profile_runs,
            field="real_sandbox_observed_prediction_error_examples",
            required_min=config.min_observed_prediction_error_examples,
            source="profile_runs",
        ),
    }
    passed = all(
        isinstance(check, dict) and check.get("passed") is True
        for check in checks.values()
    )
    report: dict[str, object] = {
        "artifact_family": "reflexcore_v0_runtime_evidence_audit",
        "passed": passed,
        "verdict": (
            "bounded_reflexcore_v0_runtime_evidence_ready"
            if passed
            else "repair_reflexcore_v0_runtime_evidence"
        ),
        "config": _json_config(config),
        "observed_summary": _observed_summary(profile_runs),
        "checks": checks,
        "claim_boundary": (
            "Audits compact real-sandbox runtime observation evidence from a "
            "ReflexCore V0 profile matrix. It supports only bounded "
            "terminal/process/filesystem/time sensory feedback and prediction-error "
            "availability; it does not prove GUI, free-shell, or production autonomy."
        ),
    }
    if config.output_json is not None:
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return report


def _observed_summary(profile_runs: list[dict[str, object]]) -> dict[str, object]:
    return {
        "profile_runs": len(profile_runs),
        "seeds": sorted(
            {int(seed) for seed in _numbers(row.get("seed") for row in profile_runs)}
        ),
        "profiles": sorted(
            str(row.get("eval_profile"))
            for row in profile_runs
            if row.get("eval_profile") is not None
        ),
        "live_episode_count_min": _min_field(
            profile_runs,
            "real_sandbox_live_episode_count",
        ),
        "runtime_observation_steps_min": _min_field(
            profile_runs,
            "real_sandbox_runtime_observation_steps",
        ),
        "changed_file_observation_steps_min": _min_field(
            profile_runs,
            "real_sandbox_changed_file_observation_steps",
        ),
        "terminal_observation_steps_min": _min_field(
            profile_runs,
            "real_sandbox_terminal_observation_steps",
        ),
        "observed_prediction_error_examples_min": _min_field(
            profile_runs,
            "real_sandbox_observed_prediction_error_examples",
        ),
        "observed_prediction_error_mean_min": _min_field(
            profile_runs,
            "real_sandbox_observed_prediction_error_mean",
        ),
        "observed_prediction_error_max_min": _min_field(
            profile_runs,
            "real_sandbox_observed_prediction_error_max",
        ),
    }


def _json_config(config: ReflexCoreRuntimeEvidenceAuditConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["matrix_report_json"] = _path_label(config.matrix_report_json)
    payload["output_json"] = _path_label(config.output_json) if config.output_json else None
    return payload


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _path_label(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _object(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _numbers(values: object) -> list[float]:
    return [number for number in (_number(value) for value in values) if number is not None]


def _min_field(rows: list[dict[str, object]], field: str) -> float | None:
    values = _numbers(row.get(field) for row in rows)
    return min(values) if values else None


def _bool_check(observed: bool, *, required: bool, source: str) -> dict[str, object]:
    return {
        "passed": observed is required,
        "observed": observed,
        "required": required,
        "source": source,
    }


def _min_check(
    observed: float | int | None,
    required_min: float | int,
    *,
    source: str,
) -> dict[str, object]:
    return {
        "passed": observed is not None and observed >= required_min,
        "observed": observed,
        "required_min": required_min,
        "source": source,
    }


def _all_rows_check(
    rows: list[dict[str, object]],
    *,
    field: str,
    required: bool,
    enabled: bool,
    source: str,
) -> dict[str, object]:
    if not enabled:
        return {
            "passed": True,
            "observed": None,
            "required": "disabled",
            "source": source,
        }
    failing = [
        _row_identity(row)
        for row in rows
        if bool(row.get(field)) is not required
    ]
    return {
        "passed": not failing,
        "observed": {"failing_rows": failing, "row_count": len(rows)},
        "required": {field: required},
        "source": source,
    }


def _row_min_check(
    rows: list[dict[str, object]],
    *,
    field: str,
    required_min: float | int,
    source: str,
) -> dict[str, object]:
    values = _numbers(row.get(field) for row in rows)
    failing = [
        {**_row_identity(row), field: row.get(field)}
        for row in rows
        if (_number(row.get(field)) is None or _number(row.get(field)) < required_min)
    ]
    return {
        "passed": bool(values) and not failing,
        "observed": {
            "min": min(values) if values else None,
            "failing_rows": failing,
        },
        "required_min": required_min,
        "source": source,
    }


def _row_identity(row: dict[str, object]) -> dict[str, object]:
    return {
        "seed": row.get("seed"),
        "train_profile": row.get("train_profile"),
        "eval_profile": row.get("eval_profile"),
    }
