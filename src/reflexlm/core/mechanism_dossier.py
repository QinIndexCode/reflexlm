from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CLAIM_BOUNDARY = (
    "Supports bounded ReflexCore V0 evidence for a 20-100M local "
    "computer-native sensory-motor core over terminal/process/filesystem/time "
    "observations with typed motor heads and allowlisted RUN_COMMAND. It does "
    "not support GUI control, vision, unrestricted shell generation, robotics, "
    "production autonomy, consciousness, AGI, or a full LLM replacement claim."
)

UNSUPPORTED_CLAIMS = [
    "GUI or visual desktop operation",
    "unrestricted shell generation",
    "production autonomous software engineering",
    "robotics or physical-world control",
    "consciousness, AGI, or human-like cognition",
    "full replacement of large language models",
]


@dataclass(slots=True)
class ReflexCoreMechanismDossierConfig:
    accepted_rollup_json: Path
    sensory_ablation_json: Path
    output_json: Path | None = None
    architecture_audit_json: Path | None = None
    negative_control_jsons: tuple[Path, ...] = ()
    min_parameter_count: int = 20_000_000
    max_parameter_count: int = 100_000_000
    min_pass_rate: float = 1.0
    min_profile_pass_rate: float = 1.0
    min_raw_action_accuracy: float = 0.85
    min_safety_gated_action_accuracy: float = 0.70
    min_offline_action_margin: float = 0.20
    min_closed_loop_success_rate: float = 0.60
    min_closed_loop_margin: float = 0.30
    min_real_sandbox_success_rate: float = 1.0
    min_real_sandbox_margin: float = 0.50
    min_next_state_relative_improvement: float = 0.30
    min_prediction_error_relative_improvement: float = 0.30
    required_ablation_modes: tuple[str, ...] = ("zero_numeric",)
    min_sensory_action_drop: float = 0.50
    min_sensory_world_drop: float = 1.0
    min_sensory_rows: int = 9
    required_seeds: tuple[int, ...] = (13, 17, 23)
    required_profiles: tuple[str, ...] = ("default", "hard", "wide_ood")


def build_reflexcore_mechanism_dossier(
    config: ReflexCoreMechanismDossierConfig,
) -> dict[str, object]:
    _validate_config(config)
    rollup = _read_json(config.accepted_rollup_json)
    sensory = _read_json(config.sensory_ablation_json)
    architecture_audit = (
        _read_json(config.architecture_audit_json)
        if config.architecture_audit_json is not None
        else None
    )
    negative_controls = [
        (path, _read_json(path)) for path in config.negative_control_jsons
    ]

    rollup_summary = _object(rollup.get("summary"))
    sensory_summary = _object(sensory.get("summary"))
    checks: dict[str, object] = {}
    checks["architecture_audit_passed"] = _architecture_audit_check(
        architecture_audit,
    )
    checks.update(_rollup_checks(rollup_summary, config, source="accepted_rollup"))
    checks.update(_sensory_checks(sensory, sensory_summary, config))
    checks["negative_controls_rejected"] = _negative_control_check(
        negative_controls,
        config,
    )

    passed = all(
        isinstance(check, dict) and check.get("passed") is True
        for check in checks.values()
    )
    source_integrity = _source_artifact_integrity(config)
    report: dict[str, object] = {
        "artifact_family": "reflexcore_v0_mechanism_dossier",
        "passed": passed,
        "verdict": (
            "bounded_reflexcore_v0_mechanism_evidence_ready"
            if passed
            else "repair_reflexcore_v0_mechanism_evidence"
        ),
        "config": _json_config(config),
        "source_artifacts": {
            "architecture_audit_json": (
                _path_label(config.architecture_audit_json)
                if config.architecture_audit_json
                else None
            ),
            "accepted_rollup_json": _path_label(config.accepted_rollup_json),
            "sensory_ablation_json": _path_label(config.sensory_ablation_json),
            "negative_control_jsons": [
                _path_label(path) for path in config.negative_control_jsons
            ],
        },
        "source_artifact_integrity": source_integrity,
        "reproducibility_fingerprint": _reproducibility_fingerprint(
            config,
            source_integrity,
        ),
        "observed_summary": _observed_summary(rollup_summary, sensory_summary),
        "checks": checks,
        "claim_boundary": CLAIM_BOUNDARY,
        "unsupported_claims": UNSUPPORTED_CLAIMS,
    }

    if config.output_json is not None:
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return report


def verify_reflexcore_mechanism_dossier(
    dossier_json: Path,
    *,
    base_dir: Path | None = None,
) -> dict[str, object]:
    dossier = _read_json(dossier_json)
    base_path = base_dir if base_dir is not None else Path.cwd()
    source_integrity = _object(dossier.get("source_artifact_integrity"))
    source_checks = _verify_source_artifact_integrity(source_integrity, base_path)
    generator_check = _verify_generator_integrity(dossier)
    fingerprint_check = _verify_reproducibility_fingerprint(dossier)
    checks = {
        "dossier_passed": _bool_check(
            dossier.get("passed") is True,
            required=True,
            source="mechanism_dossier",
        ),
        "source_artifact_integrity": source_checks,
        "generator_integrity": generator_check,
        "reproducibility_fingerprint": fingerprint_check,
    }
    passed = all(
        isinstance(check, dict) and check.get("passed") is True
        for check in checks.values()
    )
    return {
        "artifact_family": "reflexcore_v0_mechanism_dossier_verification",
        "dossier_json": _path_label(dossier_json),
        "passed": passed,
        "verdict": (
            "bounded_reflexcore_v0_mechanism_dossier_verified"
            if passed
            else "repair_reflexcore_v0_mechanism_dossier_verification"
        ),
        "checks": checks,
        "claim_boundary": (
            "Verifies dossier integrity against local source artifacts and the "
            "current mechanism_dossier generator. It does not rerun model "
            "training or benchmark experiments."
        ),
    }


def _rollup_checks(
    summary: dict[str, object],
    config: ReflexCoreMechanismDossierConfig,
    *,
    source: str,
) -> dict[str, object]:
    parameter_min = _number(summary.get("parameter_count_min"))
    parameter_max = _number(summary.get("parameter_count_max"))
    raw_action_min = _number(summary.get("raw_action_accuracy_min"))
    prompt_offline_max = _number(summary.get("prompt_only_offline_action_accuracy_max"))
    closed_loop_min = _number(summary.get("closed_loop_success_rate_min"))
    prompt_closed_loop_max = _number(
        summary.get("prompt_only_closed_loop_success_rate_max")
    )
    real_sandbox_min = _number(summary.get("real_sandbox_success_rate_min"))
    prompt_real_sandbox_max = _number(
        summary.get("prompt_only_real_sandbox_success_rate_max")
    )
    offline_margin = _margin(raw_action_min, prompt_offline_max)
    closed_loop_margin = _margin(closed_loop_min, prompt_closed_loop_max)
    real_sandbox_margin = _margin(real_sandbox_min, prompt_real_sandbox_max)
    checks = {
        "rollup_matrix_passed": _bool_check(
            bool(summary.get("matrix_passed")),
            required=True,
            source=source,
        ),
        "rollup_pass_rate": _min_check(
            _number(summary.get("pass_rate")),
            config.min_pass_rate,
            source=source,
        ),
        "rollup_profile_pass_rate": _min_check(
            _number(summary.get("profile_pass_rate")),
            config.min_profile_pass_rate,
            source=source,
        ),
        "parameter_count_in_local_range": _range_check(
            parameter_min,
            parameter_max,
            config.min_parameter_count,
            config.max_parameter_count,
            source=source,
        ),
        "raw_action_accuracy_floor": _min_check(
            raw_action_min,
            config.min_raw_action_accuracy,
            source=source,
        ),
        "safety_gated_action_accuracy_floor": _min_check(
            _number(summary.get("safety_gated_action_accuracy_min")),
            config.min_safety_gated_action_accuracy,
            source=source,
        ),
        "offline_prompt_only_margin": _min_check(
            offline_margin,
            config.min_offline_action_margin,
            source=source,
        ),
        "closed_loop_success_floor": _min_check(
            closed_loop_min,
            config.min_closed_loop_success_rate,
            source=source,
        ),
        "closed_loop_prompt_only_margin": _min_check(
            closed_loop_margin,
            config.min_closed_loop_margin,
            source=source,
        ),
        "real_sandbox_success_floor": _min_check(
            real_sandbox_min,
            config.min_real_sandbox_success_rate,
            source=source,
        ),
        "real_sandbox_prompt_only_margin": _min_check(
            real_sandbox_margin,
            config.min_real_sandbox_margin,
            source=source,
        ),
        "world_model_relative_improvement_floor": _min_check(
            _number(summary.get("next_state_relative_improvement_min")),
            config.min_next_state_relative_improvement,
            source=source,
        ),
        "prediction_error_relative_improvement_floor": _min_check(
            _number(summary.get("prediction_error_relative_improvement_min")),
            config.min_prediction_error_relative_improvement,
            source=source,
        ),
    }
    for mode in config.required_ablation_modes:
        checks[f"rollup_{mode}_action_drop_floor"] = _min_check(
            _number(summary.get(f"{mode}_action_drop_min")),
            config.min_sensory_action_drop,
            source=source,
        )
        checks[f"rollup_{mode}_world_drop_floor"] = _min_check(
            _number(summary.get(f"{mode}_world_drop_min")),
            config.min_sensory_world_drop,
            source=source,
        )
    return checks


def _sensory_checks(
    sensory: dict[str, object],
    summary: dict[str, object],
    config: ReflexCoreMechanismDossierConfig,
) -> dict[str, object]:
    rows = [_object(row) for row in _list(sensory.get("rows"))]
    observed_seeds = {int(seed) for seed in _numbers(row.get("seed") for row in rows)}
    observed_profiles = {
        str(row.get("profile")) for row in rows if row.get("profile") is not None
    }
    checks = {
        "sensory_matrix_passed": _bool_check(
            bool(sensory.get("passed")) and bool(summary.get("passed")),
            required=True,
            source="sensory_ablation_matrix",
        ),
        "sensory_row_count": _min_check(
            _number(summary.get("row_count")),
            float(config.min_sensory_rows),
            source="sensory_ablation_matrix",
        ),
        "sensory_all_rows_passed": _equality_check(
            summary.get("passed_rows"),
            summary.get("row_count"),
            source="sensory_ablation_matrix",
        ),
        "sensory_seed_coverage": _contains_check(
            observed=observed_seeds,
            required=set(config.required_seeds),
            source="sensory_ablation_matrix",
        ),
        "sensory_profile_coverage": _contains_check(
            observed=observed_profiles,
            required=set(config.required_profiles),
            source="sensory_ablation_matrix",
        ),
    }
    modes = _object(summary.get("modes"))
    for mode in config.required_ablation_modes:
        mode_summary = _object(modes.get(mode))
        checks[f"sensory_{mode}_present"] = _bool_check(
            bool(mode_summary),
            required=True,
            source="sensory_ablation_matrix",
        )
        checks[f"sensory_{mode}_action_drop_floor"] = _min_check(
            _number(_object(mode_summary.get("action_accuracy_drop")).get("min")),
            config.min_sensory_action_drop,
            source="sensory_ablation_matrix",
        )
        checks[f"sensory_{mode}_world_drop_floor"] = _min_check(
            _number(
                _object(
                    mode_summary.get("next_state_relative_improvement_drop")
                ).get("min")
            ),
            config.min_sensory_world_drop,
            source="sensory_ablation_matrix",
        )
    return checks


def _negative_control_check(
    negative_controls: list[tuple[Path, dict[str, object]]],
    config: ReflexCoreMechanismDossierConfig,
) -> dict[str, object]:
    details: list[dict[str, object]] = []
    for path, payload in negative_controls:
        summary = _object(payload.get("summary"))
        control_checks = _rollup_checks(summary, config, source="negative_control")
        accepted = all(
            isinstance(check, dict) and check.get("passed") is True
            for check in control_checks.values()
        )
        details.append(
            {
                "path": _path_label(path),
                "accepted_by_primary_rollup_gate": accepted,
                "matrix_passed": bool(summary.get("matrix_passed")),
                "profile_pass_rate": summary.get("profile_pass_rate"),
                "pass_rate": summary.get("pass_rate"),
                "real_sandbox_success_rate_min": summary.get(
                    "real_sandbox_success_rate_min"
                ),
            }
        )
    return {
        "passed": all(not detail["accepted_by_primary_rollup_gate"] for detail in details),
        "observed": details,
        "required": "each negative control must fail the primary rollup gate",
        "source": "negative_controls",
    }


def _architecture_audit_check(
    architecture_audit: dict[str, object] | None,
) -> dict[str, object]:
    if architecture_audit is None:
        return {
            "passed": True,
            "observed": None,
            "required": "optional architecture audit not provided",
            "source": "architecture_audit",
        }
    return {
        "passed": architecture_audit.get("passed") is True,
        "observed": {
            "artifact_family": architecture_audit.get("artifact_family"),
            "verdict": architecture_audit.get("verdict"),
            "passed": architecture_audit.get("passed"),
        },
        "required": "architecture audit must pass when provided",
        "source": "architecture_audit",
    }


def _observed_summary(
    rollup_summary: dict[str, object],
    sensory_summary: dict[str, object],
) -> dict[str, object]:
    return {
        "run_count": rollup_summary.get("runs"),
        "parameter_count_min": rollup_summary.get("parameter_count_min"),
        "parameter_count_max": rollup_summary.get("parameter_count_max"),
        "pass_rate": rollup_summary.get("pass_rate"),
        "profile_pass_rate": rollup_summary.get("profile_pass_rate"),
        "raw_action_accuracy_min": rollup_summary.get("raw_action_accuracy_min"),
        "safety_gated_action_accuracy_min": rollup_summary.get(
            "safety_gated_action_accuracy_min"
        ),
        "prompt_only_offline_action_accuracy_max": rollup_summary.get(
            "prompt_only_offline_action_accuracy_max"
        ),
        "closed_loop_success_rate_min": rollup_summary.get(
            "closed_loop_success_rate_min"
        ),
        "prompt_only_closed_loop_success_rate_max": rollup_summary.get(
            "prompt_only_closed_loop_success_rate_max"
        ),
        "real_sandbox_success_rate_min": rollup_summary.get(
            "real_sandbox_success_rate_min"
        ),
        "prompt_only_real_sandbox_success_rate_max": rollup_summary.get(
            "prompt_only_real_sandbox_success_rate_max"
        ),
        "next_state_relative_improvement_min": rollup_summary.get(
            "next_state_relative_improvement_min"
        ),
        "prediction_error_relative_improvement_min": rollup_summary.get(
            "prediction_error_relative_improvement_min"
        ),
        "sensory_rows": sensory_summary.get("row_count"),
        "sensory_passed_rows": sensory_summary.get("passed_rows"),
        "sensory_modes": sensory_summary.get("modes"),
    }


def _validate_config(config: ReflexCoreMechanismDossierConfig) -> None:
    if not config.accepted_rollup_json.exists():
        raise FileNotFoundError(
            f"accepted_rollup_json does not exist: {config.accepted_rollup_json}"
        )
    if not config.sensory_ablation_json.exists():
        raise FileNotFoundError(
            f"sensory_ablation_json does not exist: {config.sensory_ablation_json}"
        )
    if config.architecture_audit_json is not None and not config.architecture_audit_json.exists():
        raise FileNotFoundError(
            f"architecture_audit_json does not exist: {config.architecture_audit_json}"
        )
    for path in config.negative_control_jsons:
        if not path.exists():
            raise FileNotFoundError(f"negative_control_json does not exist: {path}")
    if config.min_parameter_count <= 0:
        raise ValueError("min_parameter_count must be positive")
    if config.max_parameter_count < config.min_parameter_count:
        raise ValueError("max_parameter_count must be >= min_parameter_count")
    if not config.required_ablation_modes:
        raise ValueError("at least one required ablation mode is required")
    if config.min_sensory_rows < 1:
        raise ValueError("min_sensory_rows must be >= 1")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _json_config(config: ReflexCoreMechanismDossierConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["accepted_rollup_json"] = _path_label(config.accepted_rollup_json)
    payload["sensory_ablation_json"] = _path_label(config.sensory_ablation_json)
    payload["output_json"] = _path_label(config.output_json) if config.output_json else None
    payload["architecture_audit_json"] = (
        _path_label(config.architecture_audit_json)
        if config.architecture_audit_json
        else None
    )
    payload["negative_control_jsons"] = [
        _path_label(path) for path in config.negative_control_jsons
    ]
    payload["required_ablation_modes"] = list(config.required_ablation_modes)
    payload["required_seeds"] = list(config.required_seeds)
    payload["required_profiles"] = list(config.required_profiles)
    return payload


def _source_artifact_integrity(
    config: ReflexCoreMechanismDossierConfig,
) -> dict[str, object]:
    return {
        "architecture_audit_json": (
            _artifact_metadata(config.architecture_audit_json)
            if config.architecture_audit_json
            else None
        ),
        "accepted_rollup_json": _artifact_metadata(config.accepted_rollup_json),
        "sensory_ablation_json": _artifact_metadata(config.sensory_ablation_json),
        "negative_control_jsons": [
            _artifact_metadata(path) for path in config.negative_control_jsons
        ],
    }


def _artifact_metadata(path: Path) -> dict[str, object]:
    return {
        "path": _path_label(path),
        "size_bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source_artifact_integrity(
    source_integrity: dict[str, object],
    base_dir: Path,
) -> dict[str, object]:
    observed = {
        "architecture_audit_json": _verify_artifact_metadata(
            source_integrity.get("architecture_audit_json"),
            base_dir,
        ),
        "accepted_rollup_json": _verify_artifact_metadata(
            source_integrity.get("accepted_rollup_json"),
            base_dir,
        ),
        "sensory_ablation_json": _verify_artifact_metadata(
            source_integrity.get("sensory_ablation_json"),
            base_dir,
        ),
        "negative_control_jsons": [
            _verify_artifact_metadata(item, base_dir)
            for item in _list(source_integrity.get("negative_control_jsons"))
        ],
    }
    required = [
        observed["accepted_rollup_json"],
        observed["sensory_ablation_json"],
        *observed["negative_control_jsons"],
    ]
    if observed["architecture_audit_json"] is not None:
        required.append(observed["architecture_audit_json"])
    return {
        "passed": all(
            isinstance(item, dict) and item.get("passed") is True for item in required
        ),
        "observed": observed,
        "required": "recorded source artifact size and sha256 must match local files",
        "source": "source_artifact_integrity",
    }


def _verify_artifact_metadata(
    metadata: object,
    base_dir: Path,
) -> dict[str, object] | None:
    if metadata is None:
        return None
    metadata_obj = _object(metadata)
    path_value = metadata_obj.get("path")
    expected_size = metadata_obj.get("size_bytes")
    expected_sha256 = metadata_obj.get("sha256")
    if not isinstance(path_value, str):
        return {
            "passed": False,
            "path": None,
            "reason": "missing artifact path",
        }
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        return {
            "passed": False,
            "path": path_value,
            "reason": "artifact missing",
        }
    observed_size = path.stat().st_size
    observed_sha256 = _file_sha256(path)
    return {
        "passed": observed_size == expected_size and observed_sha256 == expected_sha256,
        "path": path_value,
        "observed_size_bytes": observed_size,
        "expected_size_bytes": expected_size,
        "observed_sha256": observed_sha256,
        "expected_sha256": expected_sha256,
    }


def _verify_generator_integrity(dossier: dict[str, object]) -> dict[str, object]:
    fingerprint = _object(dossier.get("reproducibility_fingerprint"))
    generator = _object(fingerprint.get("generator"))
    expected_sha256 = generator.get("sha256")
    observed_sha256 = _file_sha256(Path(__file__))
    return {
        "passed": observed_sha256 == expected_sha256,
        "observed": {
            "path": Path(__file__).name,
            "sha256": observed_sha256,
        },
        "required": generator,
        "source": "reproducibility_fingerprint.generator",
    }


def _verify_reproducibility_fingerprint(
    dossier: dict[str, object],
) -> dict[str, object]:
    fingerprint = _object(dossier.get("reproducibility_fingerprint"))
    payload = {
        "config": _object(dossier.get("config")),
        "environment": _object(fingerprint.get("environment")),
        "generator": _object(fingerprint.get("generator")),
        "source_artifact_integrity": _object(dossier.get("source_artifact_integrity")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    observed_sha256 = hashlib.sha256(encoded).hexdigest()
    expected_sha256 = fingerprint.get("sha256")
    return {
        "passed": observed_sha256 == expected_sha256,
        "observed_sha256": observed_sha256,
        "expected_sha256": expected_sha256,
        "source": "reproducibility_fingerprint",
    }


def _reproducibility_fingerprint(
    config: ReflexCoreMechanismDossierConfig,
    source_integrity: dict[str, object],
) -> dict[str, object]:
    generator = {
        "path": Path(__file__).name,
        "sha256": _file_sha256(Path(__file__)),
    }
    environment = {
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
    }
    payload = {
        "config": _json_config(config),
        "environment": environment,
        "generator": generator,
        "source_artifact_integrity": source_integrity,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "generator": generator,
        "environment": environment,
        "canonical_fields": [
            "config",
            "environment",
            "generator",
            "source_artifact_integrity",
        ],
    }


def _path_label(path: Path) -> str:
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
    number = float(value)
    return number if math.isfinite(number) else None


def _numbers(values: object) -> list[float]:
    return [number for number in (_number(value) for value in values) if number is not None]


def _margin(model_value: float | None, baseline_value: float | None) -> float | None:
    if model_value is None or baseline_value is None:
        return None
    return model_value - baseline_value


def _bool_check(observed: bool, *, required: bool, source: str) -> dict[str, object]:
    return {
        "passed": observed is required,
        "observed": observed,
        "required": required,
        "source": source,
    }


def _min_check(
    observed: float | None,
    required_min: float,
    *,
    source: str,
) -> dict[str, object]:
    return {
        "passed": observed is not None and observed >= required_min,
        "observed": observed,
        "required_min": required_min,
        "source": source,
    }


def _range_check(
    observed_min: float | None,
    observed_max: float | None,
    required_min: float,
    required_max: float,
    *,
    source: str,
) -> dict[str, object]:
    passed = (
        observed_min is not None
        and observed_max is not None
        and observed_min >= required_min
        and observed_max <= required_max
    )
    return {
        "passed": passed,
        "observed": {"min": observed_min, "max": observed_max},
        "required": {"min": required_min, "max": required_max},
        "source": source,
    }


def _equality_check(
    observed: object,
    required: object,
    *,
    source: str,
) -> dict[str, object]:
    return {
        "passed": observed == required and observed is not None,
        "observed": observed,
        "required": required,
        "source": source,
    }


def _contains_check(
    *,
    observed: set[object],
    required: set[object],
    source: str,
) -> dict[str, object]:
    missing = sorted(required - observed, key=str)
    return {
        "passed": not missing,
        "observed": sorted(observed, key=str),
        "required": sorted(required, key=str),
        "missing": missing,
        "source": source,
    }
