import json
from pathlib import Path

from reflexlm.cli.build_phase2aa_runtime_replication_report import (
    build_phase2aa_runtime_replication_report,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _data_health() -> dict:
    return {
        "passed": True,
        "checks": {"required_runtime_artifacts_available": True},
    }


def _delta_gate(*, passed: bool = True) -> dict:
    return {
        "passed": passed,
        "metrics": {"full_minus_control_success_rate": 0.6},
    }


def _rows(
    *,
    count: int = 256,
    policy_loaded: bool = True,
    success_rate: float = 1.0,
    policy_label: str = "policy",
) -> list[dict]:
    successes = int(count * success_rate)
    return [
        {
            "trace_id": f"holdout:repo:{index}",
            "native_policy_label": policy_label,
            "policy_loaded": policy_loaded,
            "success": index < successes,
            "patch_candidate_selected_correctly": index < successes,
            "claim_boundary": "bounded_patch_candidate_selection_not_freeform_patch_generation",
        }
        for index in range(count)
    ]


def test_phase2aa_runtime_replication_report_accepts_cross_model_delta(
    tmp_path: Path,
) -> None:
    report = build_phase2aa_runtime_replication_report(
        data_health_json=_write_json(tmp_path / "data.json", _data_health()),
        full_delta_gate_jsons=[
            _write_json(tmp_path / "delta_1.json", _delta_gate()),
            _write_json(tmp_path / "delta_2.json", _delta_gate()),
            _write_json(tmp_path / "delta_3.json", _delta_gate()),
            _write_json(tmp_path / "delta_4.json", _delta_gate()),
        ],
        full_execution_jsonls=[
            _write_jsonl(tmp_path / "full_1.jsonl", _rows(policy_label="p1")),
            _write_jsonl(tmp_path / "full_2.jsonl", _rows(policy_label="p2_seed13")),
            _write_jsonl(tmp_path / "full_3.jsonl", _rows(policy_label="p2_seed17")),
            _write_jsonl(tmp_path / "full_4.jsonl", _rows(policy_label="p2_seed23")),
        ],
        full_package_jsons=[
            _write_json(tmp_path / "pkg_1.json", {"base_model_name": "model-1"}),
            _write_json(tmp_path / "pkg_2.json", {"base_model_name": "model-2"}),
            _write_json(tmp_path / "pkg_3.json", {"base_model_name": "model-2"}),
            _write_json(tmp_path / "pkg_4.json", {"base_model_name": "model-2"}),
        ],
        control_execution_jsonl=_write_jsonl(
            tmp_path / "control.jsonl",
            _rows(policy_loaded=False, success_rate=0.4, policy_label="control"),
        ),
        no_nsi_execution_jsonl=_write_jsonl(
            tmp_path / "no_nsi.jsonl",
            _rows(success_rate=0.4, policy_label="no_nsi"),
        ),
        symbolic_runtime_delta_gate_json=_write_json(
            tmp_path / "symbolic.json", {"passed": False, "metrics": {}}
        ),
    )

    assert report["passed"] is True
    assert report["metrics"]["model_count"] == 2
    assert report["metrics"]["max_seed_count_for_model"] == 3
    assert report["metrics"]["full_minus_no_nsi_success_rate_min"] >= 0.15
    assert "independent_seed_packages_for_same_runtime_delta" not in report[
        "next_required_evidence"
    ]


def test_phase2aa_runtime_replication_report_rejects_single_model(
    tmp_path: Path,
) -> None:
    report = build_phase2aa_runtime_replication_report(
        data_health_json=_write_json(tmp_path / "data.json", _data_health()),
        full_delta_gate_jsons=[_write_json(tmp_path / "delta.json", _delta_gate())],
        full_execution_jsonls=[
            _write_jsonl(tmp_path / "full.jsonl", _rows(policy_label="p1")),
        ],
        full_package_jsons=[
            _write_json(tmp_path / "pkg.json", {"base_model_name": "model-1"}),
        ],
        control_execution_jsonl=_write_jsonl(
            tmp_path / "control.jsonl",
            _rows(policy_loaded=False, success_rate=0.4, policy_label="control"),
        ),
        no_nsi_execution_jsonl=_write_jsonl(
            tmp_path / "no_nsi.jsonl",
            _rows(success_rate=0.4, policy_label="no_nsi"),
        ),
        symbolic_runtime_delta_gate_json=_write_json(
            tmp_path / "symbolic.json", {"passed": False, "metrics": {}}
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["model_count_minimum_met"] is False


def test_phase2aa_runtime_replication_report_rejects_missing_independent_seeds(
    tmp_path: Path,
) -> None:
    report = build_phase2aa_runtime_replication_report(
        data_health_json=_write_json(tmp_path / "data.json", _data_health()),
        full_delta_gate_jsons=[
            _write_json(tmp_path / "delta_1.json", _delta_gate()),
            _write_json(tmp_path / "delta_2.json", _delta_gate()),
        ],
        full_execution_jsonls=[
            _write_jsonl(tmp_path / "full_1.jsonl", _rows(policy_label="p1")),
            _write_jsonl(tmp_path / "full_2.jsonl", _rows(policy_label="p2_seed13")),
        ],
        full_package_jsons=[
            _write_json(tmp_path / "pkg_1.json", {"base_model_name": "model-1"}),
            _write_json(tmp_path / "pkg_2.json", {"base_model_name": "model-2"}),
        ],
        control_execution_jsonl=_write_jsonl(
            tmp_path / "control.jsonl",
            _rows(policy_loaded=False, success_rate=0.4, policy_label="control"),
        ),
        no_nsi_execution_jsonl=_write_jsonl(
            tmp_path / "no_nsi.jsonl",
            _rows(success_rate=0.4, policy_label="no_nsi"),
        ),
        symbolic_runtime_delta_gate_json=_write_json(
            tmp_path / "symbolic.json", {"passed": False, "metrics": {}}
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["independent_seed_count_minimum_met"] is False
    assert "independent_seed_packages_for_same_runtime_delta" in report[
        "next_required_evidence"
    ]


def test_phase2aa_runtime_replication_report_rejects_no_nsi_tie(
    tmp_path: Path,
) -> None:
    report = build_phase2aa_runtime_replication_report(
        data_health_json=_write_json(tmp_path / "data.json", _data_health()),
        full_delta_gate_jsons=[
            _write_json(tmp_path / "delta_1.json", _delta_gate()),
            _write_json(tmp_path / "delta_2.json", _delta_gate()),
        ],
        full_execution_jsonls=[
            _write_jsonl(tmp_path / "full_1.jsonl", _rows(policy_label="p1")),
            _write_jsonl(tmp_path / "full_2.jsonl", _rows(policy_label="p2")),
        ],
        full_package_jsons=[
            _write_json(tmp_path / "pkg_1.json", {"base_model_name": "model-1"}),
            _write_json(tmp_path / "pkg_2.json", {"base_model_name": "model-2"}),
        ],
        control_execution_jsonl=_write_jsonl(
            tmp_path / "control.jsonl",
            _rows(policy_loaded=False, success_rate=0.4, policy_label="control"),
        ),
        no_nsi_execution_jsonl=_write_jsonl(
            tmp_path / "no_nsi.jsonl",
            _rows(success_rate=1.0, policy_label="no_nsi"),
        ),
        symbolic_runtime_delta_gate_json=_write_json(
            tmp_path / "symbolic.json", {"passed": False, "metrics": {}}
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["full_minus_no_nsi_delta_met"] is False


def test_phase2aa_runtime_replication_report_can_require_retry_control_delta(
    tmp_path: Path,
) -> None:
    report = build_phase2aa_runtime_replication_report(
        data_health_json=_write_json(tmp_path / "data.json", _data_health()),
        full_delta_gate_jsons=[
            _write_json(tmp_path / "delta_1.json", _delta_gate()),
            _write_json(tmp_path / "delta_2.json", _delta_gate()),
            _write_json(tmp_path / "delta_3.json", _delta_gate()),
            _write_json(tmp_path / "delta_4.json", _delta_gate()),
        ],
        full_execution_jsonls=[
            _write_jsonl(tmp_path / "full_1.jsonl", _rows(policy_label="p1")),
            _write_jsonl(tmp_path / "full_2.jsonl", _rows(policy_label="p2_seed13")),
            _write_jsonl(tmp_path / "full_3.jsonl", _rows(policy_label="p2_seed17")),
            _write_jsonl(tmp_path / "full_4.jsonl", _rows(policy_label="p2_seed23")),
        ],
        full_package_jsons=[
            _write_json(tmp_path / "pkg_1.json", {"base_model_name": "model-1"}),
            _write_json(tmp_path / "pkg_2.json", {"base_model_name": "model-2"}),
            _write_json(tmp_path / "pkg_3.json", {"base_model_name": "model-2"}),
            _write_json(tmp_path / "pkg_4.json", {"base_model_name": "model-2"}),
        ],
        control_execution_jsonl=_write_jsonl(
            tmp_path / "control.jsonl",
            _rows(policy_loaded=False, success_rate=0.4, policy_label="control"),
        ),
        no_nsi_execution_jsonl=_write_jsonl(
            tmp_path / "no_nsi.jsonl",
            _rows(success_rate=0.4, policy_label="no_nsi"),
        ),
        retry_control_execution_jsonl=_write_jsonl(
            tmp_path / "retry.jsonl",
            _rows(policy_loaded=False, success_rate=1.0, policy_label="retry"),
        ),
        symbolic_runtime_delta_gate_json=_write_json(
            tmp_path / "symbolic.json", {"passed": False, "metrics": {}}
        ),
        require_full_minus_retry_control=True,
    )

    assert report["passed"] is False
    assert report["checks"]["full_minus_retry_control_delta_met_if_required"] is False
    assert "task_family_where_identity_first_retry_control_does_not_tie_full" in report[
        "next_required_evidence"
    ]
