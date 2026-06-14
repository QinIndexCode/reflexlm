import json
from pathlib import Path

from reflexlm.cli.audit_phase2homeostasis_publication_bundle import (
    POSITIVE_REPORT_SPECS,
    audit_phase2homeostasis_publication_bundle,
    validate_phase2homeostasis_publication_bundle,
)
from reflexlm.runtime.homeostasis import (
    HomeostaticControlConfig,
    HomeostaticSynapticController,
)
from reflexlm.schema import ActionType


AUTH_KEY = "publication-bundle-test-key"


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _positive_report(spec: dict[str, str], *, overstated: bool = False) -> dict:
    ready_flags = {
        "phase2ci_unified_package_open_task_family_repo_runtime": {
            "ready_for_bounded_generated_task_family_runtime_claim": True,
            "ready_for_unified_package_generated_task_family_runtime_claim": True,
        },
        "phase2homeostasis_persistent_state_chain": {
            "ready_for_bounded_cross_process_homeostatic_memory_claim": True,
            "ready_for_unbounded_long_term_memory_claim": False,
        },
        "phase2homeostasis_cross_runtime_dynamics": {
            "ready_for_bounded_cross_runtime_homeostatic_dynamics_claim": True,
            "ready_for_general_runtime_interpreter_invariance_claim": False,
        },
        "phase2cj_runtime_interpreter_invariance_audit": {
            "ready_for_bounded_runtime_interpreter_invariance_claim": True,
            "ready_for_general_runtime_interpreter_invariance_claim": False,
        },
    }[spec["family"]]
    return {
        "artifact_family": spec["family"],
        "passed": True,
        **ready_flags,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": overstated,
        "checks": {"local_check": True},
        "metrics": {
            "repositories": 3,
            "episodes": 18,
            "executed_actions": 114,
            "task_completion_success_rate": 1.0,
            "maximum_active_threshold_delta": 0.0001,
            "episodes_per_runtime": 18,
        },
        "next_required_experiment": "fresh_directory_replay",
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


def _fixture_paths(tmp_path: Path) -> dict[str, Path | list[Path]]:
    reports = [
        _write(
            tmp_path / f"{spec['role']}.json",
            _positive_report(spec),
        )
        for spec in POSITIVE_REPORT_SPECS
    ]
    states = [
        _write(tmp_path / f"state-{index}.json", _state_artifact())
        for index in range(3)
    ]
    negative = _write(tmp_path / "no-key-negative.json", _negative_report())
    return {
        "reports": reports,
        "states": states,
        "negative": negative,
    }


def _run_bundle(tmp_path: Path, paths: dict[str, Path | list[Path]], **kwargs):
    reports = paths["reports"]
    assert isinstance(reports, list)
    states = paths["states"]
    assert isinstance(states, list)
    return audit_phase2homeostasis_publication_bundle(
        generation1_report_json=reports[0],
        generation2_py313_report_json=reports[1],
        generation2_py312_report_json=reports[2],
        chain_py313_report_json=reports[3],
        chain_py313_to_py312_report_json=reports[4],
        cross_runtime_dynamics_report_json=reports[5],
        phase2cj_report_json=reports[6],
        no_key_negative_report_json=paths["negative"],
        state_jsons=states,
        output_report_json=tmp_path / "bundle.json",
        output_markdown=tmp_path / "bundle.md",
        authenticity_key=AUTH_KEY,
        **kwargs,
    )


def test_homeostasis_publication_bundle_accepts_hmac_evidence(
    tmp_path: Path,
) -> None:
    report = _run_bundle(tmp_path, _fixture_paths(tmp_path))
    validation = validate_phase2homeostasis_publication_bundle(report)

    assert report["passed"] is True
    assert validation["passed"] is True
    assert report["metrics"]["positive_report_count"] == len(POSITIVE_REPORT_SPECS)
    assert report["metrics"]["hmac_state_artifact_count"] == 3
    assert Path(report["evidence"]["output_markdown"]).exists()
    assert report["ready_for_epoch_making_architecture_claim"] is False


def test_homeostasis_publication_bundle_rejects_unbounded_state_payload(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    states = paths["states"]
    assert isinstance(states, list)
    artifact = json.loads(Path(states[0]).read_text(encoding="utf-8"))
    artifact["state"]["unbounded_semantic_memory"] = "not allowed"
    Path(states[0]).write_text(json.dumps(artifact), encoding="utf-8")

    report = _run_bundle(tmp_path, paths)
    validation = validate_phase2homeostasis_publication_bundle(report)

    assert report["passed"] is False
    assert validation["passed"] is False
    assert report["checks"]["all_state_artifacts_hmac_v3_bounded_and_valid"] is False


def test_homeostasis_publication_bundle_rejects_overstated_claim(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    reports = paths["reports"]
    assert isinstance(reports, list)
    _write(reports[5], _positive_report(POSITIVE_REPORT_SPECS[5], overstated=True))

    report = _run_bundle(tmp_path, paths)
    validation = validate_phase2homeostasis_publication_bundle(report)

    assert report["passed"] is False
    assert validation["passed"] is False
    assert report["checks"]["all_positive_reports_passed_and_bounded"] is False


def test_homeostasis_publication_bundle_rejects_negative_control_that_passed(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    paths["negative"] = _write(
        tmp_path / "bad-negative.json",
        _negative_report(expected_failure=False),
    )

    report = _run_bundle(tmp_path, paths)
    validation = validate_phase2homeostasis_publication_bundle(report)

    assert report["passed"] is False
    assert validation["passed"] is False
    assert (
        report["checks"]["hmac_missing_key_negative_control_failed_closed"] is False
    )


def test_homeostasis_publication_bundle_rejects_forbidden_secret_match(
    tmp_path: Path,
) -> None:
    paths = _fixture_paths(tmp_path)
    report = _run_bundle(
        tmp_path,
        paths,
        forbidden_strings=["hmac-sha256"],
    )
    validation = validate_phase2homeostasis_publication_bundle(report)

    assert report["passed"] is False
    assert validation["passed"] is False
    assert report["checks"]["forbidden_secret_scan_clean"] is False
