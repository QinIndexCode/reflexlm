import json
from pathlib import Path

from reflexlm.cli.audit_phase2homeostasis_bounded_publication_dossier import (
    CORE_REPORT_SPECS,
    audit_phase2homeostasis_bounded_publication_dossier,
    validate_phase2homeostasis_bounded_publication_dossier,
)
from reflexlm.runtime.homeostasis import (
    HomeostaticControlConfig,
    HomeostaticSynapticController,
)
from reflexlm.schema import ActionType


AUTH_KEY = "bounded-dossier-test-key"


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _ready_flags(family: str) -> dict:
    return {
        "phase2ci_unified_package_open_task_family_repo_runtime": {
            "ready_for_bounded_generated_task_family_runtime_claim": True,
        },
        "phase2homeostasis_persistent_state_chain": {
            "ready_for_bounded_cross_process_homeostatic_memory_claim": True,
        },
        "phase2cj_runtime_interpreter_invariance_audit": {
            "ready_for_bounded_runtime_interpreter_invariance_claim": True,
        },
        "phase2homeostasis_fresh_rerun_limitations": {
            "ready_for_bounded_homeostasis_fresh_behavioral_claim": True,
            "ready_for_exact_cross_runtime_homeostatic_dynamics_claim": False,
        },
    }[family]


def _core_report(spec: dict[str, str], *, overstated: bool = False) -> dict:
    metrics = {
        "repositories": 3,
        "episodes": 18,
        "executed_actions": 114,
        "task_completion_success_rate": 1.0,
    }
    if spec["family"] == "phase2homeostasis_fresh_rerun_limitations":
        metrics.update(
            {
                "threshold_drift_limit": 0.01,
                "maximum_active_threshold_delta": 0.008,
                "side_effect_trace_rows": 18,
            }
        )
    return {
        "artifact_family": spec["family"],
        "passed": True,
        **_ready_flags(spec["family"]),
        "ready_for_unbounded_long_term_memory_claim": False,
        "ready_for_general_runtime_interpreter_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": overstated,
        "checks": {"ok": True},
        "metrics": metrics,
        "evidence": (
            {"cross_runtime_dynamics_report_json": str(Path("cross-runtime.json"))}
            if spec["family"] == "phase2homeostasis_fresh_rerun_limitations"
            else {}
        ),
    }


def _state_artifact() -> dict:
    config = HomeostaticControlConfig(
        surprise_wake_threshold=0.80,
        failure_sensitivity_rate=1.0,
        preserve_adaptive_threshold_across_reset=True,
    )
    controller = HomeostaticSynapticController(config)
    controller.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.20,
        risk=0.10,
        prediction_error=0.40,
        temporal_observation_available=True,
        failure_visible=True,
    )
    controller.reset()
    return controller.export_persistent_state(authenticity_key=AUTH_KEY)


def _negative_report(*, expected_failure: bool = True) -> dict:
    return {
        "artifact_family": "phase2homeostasis_persistent_state_chain",
        "passed": not expected_failure,
        "ready_for_bounded_cross_process_homeostatic_memory_claim": False,
        "checks": {
            "both_artifact_integrities_valid": not expected_failure,
        },
        "metrics": {
            "generation1_authenticator_algorithm": "hmac-sha256",
            "generation2_authenticator_algorithm": "hmac-sha256",
        },
    }


def _cross_runtime_failure(
    *,
    passed: bool = False,
    maximum_active_threshold_delta: float = 0.008,
) -> dict:
    return {
        "artifact_family": "phase2homeostasis_cross_runtime_dynamics",
        "passed": passed,
        "ready_for_bounded_cross_runtime_homeostatic_dynamics_claim": False,
        "ready_for_general_runtime_interpreter_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": {
            "canonical_runtime_passed": True,
            "alternate_runtime_passed": True,
            "same_seed_and_task_matrix": True,
            "same_core_completion_metrics": True,
            "discrete_homeostatic_dynamics_match": False,
            "active_threshold_deltas_within_tolerance": False,
            "wake_reason_counts_match": False,
            "runtime_normalized_executable_action_traces_match": False,
        },
        "metrics": {
            "maximum_active_threshold_delta": maximum_active_threshold_delta,
        },
    }


def _fixture_paths(tmp_path: Path) -> dict[str, Path | list[Path]]:
    reports = {
        spec["role"]: _write(tmp_path / f"{spec['role']}.json", _core_report(spec))
        for spec in CORE_REPORT_SPECS
    }
    cross_runtime = _write(
        tmp_path / "cross-runtime.json",
        _cross_runtime_failure(),
    )
    limitation_path = reports["fresh_rerun_limitations"]
    limitation = json.loads(limitation_path.read_text(encoding="utf-8"))
    limitation["evidence"]["cross_runtime_dynamics_report_json"] = str(cross_runtime)
    _write(limitation_path, limitation)
    states = [
        _write(tmp_path / f"state-{index}.json", _state_artifact())
        for index in range(3)
    ]
    negative = _write(tmp_path / "no-key-negative.json", _negative_report())
    return {
        "reports": reports,
        "cross_runtime": cross_runtime,
        "states": states,
        "negative": negative,
    }


def _run_dossier(tmp_path: Path, paths: dict[str, Path | list[Path]], **kwargs):
    reports = paths["reports"]
    assert isinstance(reports, dict)
    states = paths["states"]
    assert isinstance(states, list)
    return audit_phase2homeostasis_bounded_publication_dossier(
        generation1_report_json=reports["runtime_generation1_py313"],
        generation2_py313_report_json=reports["runtime_generation2_py313"],
        generation2_py312_report_json=reports["runtime_generation2_py312"],
        chain_py313_report_json=reports["persistent_chain_py313"],
        chain_py313_to_py312_report_json=reports["persistent_chain_py313_to_py312"],
        phase2cj_report_json=reports["runtime_interpreter_invariance"],
        fresh_rerun_limitations_report_json=reports["fresh_rerun_limitations"],
        cross_runtime_dynamics_report_json=paths["cross_runtime"],
        no_key_negative_report_json=paths["negative"],
        state_jsons=states,
        output_report_json=tmp_path / "dossier.json",
        output_markdown=tmp_path / "dossier.md",
        authenticity_key=AUTH_KEY,
        **kwargs,
    )


def test_bounded_publication_dossier_accepts_limited_fresh_evidence(
    tmp_path: Path,
) -> None:
    report = _run_dossier(tmp_path, _fixture_paths(tmp_path))
    validation = validate_phase2homeostasis_bounded_publication_dossier(report)

    assert report["passed"] is True
    assert validation["passed"] is True
    assert report["ready_for_bounded_homeostasis_publication_dossier_claim"] is True
    assert report["ready_for_exact_cross_runtime_homeostatic_dynamics_claim"] is False
    assert Path(report["evidence"]["output_markdown"]).exists()


def test_bounded_publication_dossier_rejects_exact_runtime_pass_claim(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    _write(paths["cross_runtime"], _cross_runtime_failure(passed=True))
    report = _run_dossier(tmp_path, paths)
    validation = validate_phase2homeostasis_bounded_publication_dossier(report)

    assert report["passed"] is False
    assert validation["passed"] is False
    assert report["checks"]["exact_cross_runtime_limitation_recorded"] is False


def test_bounded_publication_dossier_rejects_unbounded_threshold_drift(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    _write(
        paths["cross_runtime"],
        _cross_runtime_failure(maximum_active_threshold_delta=0.02),
    )
    report = _run_dossier(tmp_path, paths)
    validation = validate_phase2homeostasis_bounded_publication_dossier(report)

    assert report["passed"] is False
    assert validation["passed"] is False
    assert report["checks"]["exact_cross_runtime_limitation_recorded"] is False


def test_bounded_publication_dossier_rejects_epoch_claim(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    reports = paths["reports"]
    assert isinstance(reports, dict)
    first_spec = CORE_REPORT_SPECS[0]
    _write(reports[first_spec["role"]], _core_report(first_spec, overstated=True))
    report = _run_dossier(tmp_path, paths)
    validation = validate_phase2homeostasis_bounded_publication_dossier(report)

    assert report["passed"] is False
    assert validation["passed"] is False
    assert report["checks"]["core_positive_reports_passed_and_bounded"] is False
