import json
from pathlib import Path

from reflexlm.cli.audit_phase2bb_package_loaded_execution_matrix import (
    audit_phase2bb_package_loaded_execution_matrix,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _subset() -> dict:
    return {
        "passed": True,
        "head_rows": 10,
        "task_rows": 10,
        "repo_count": 5,
        "repos": ["a", "b", "c", "d", "e"],
        "clone_filtered_repos": ["filtered"],
        "slot_counts": {"0": 5, "1": 5},
        "inputs": {
            "clone_root": "artifacts/external_repos/phase2av_v14_candidates",
            "require_clone_present": True,
        },
    }


def _package_gate() -> dict:
    return {
        "passed": True,
        "ready_for_phase2az_packaged_adapter_runtime_smoke": True,
        "ready_for_epoch_making_architecture_claim": False,
    }


def _execution(*, rows: int = 10, success_rate: float = 0.8) -> dict:
    return {
        "selection_policy": "package_loaded_native_head",
        "rows": rows,
        "slot_selection_accuracy": 1.0,
        "execution_attempts": rows,
        "success_rate": success_rate,
        "attempt_success_rate": success_rate,
        "recorded_patch_artifact_used_rows": 0,
        "recorded_patch_artifact_used_for_fault_injection_rows": rows,
        "claim_bearing_execution_evidence_rows": rows,
        "freeform_patch_generation_rows": 0,
        "sealed_feedback_used_rows": 0,
        "model_prediction_records_present_rows": 0,
        "package_policy_loaded_rows": rows,
        "package_model_load_strategy": "single_device",
        "package_offload_state_dict": True,
        "test_python_profiles": {"map_default": rows},
        "package_qwen_called_rows": rows,
        "package_open_repair_authorized_rows": rows,
        "package_head_record_visible_state_rows": rows,
    }


def _wrong_cache() -> dict:
    return {
        "selection_policy": "wrong_cache",
        "rows": 10,
        "execution_attempts": 0,
        "success_rate": 0.0,
    }


def _execution_rows() -> list[dict]:
    return [
        {
            "success": True,
            "phase2z_symbolic_patch_failure": None,
            "stop_condition": "verification_passed",
            "package_policy_metadata": {
                "model_load_strategy": "single_device",
                "offload_state_dict": True,
            },
            "test_python": "python",
            "test_python_source": "map_default",
        }
        for _ in range(8)
    ] + [
        {
            "success": False,
            "phase2z_symbolic_patch_failure": "missing_symbolic_structural_requirements",
            "stop_condition": "verification_failed_stop",
            "package_policy_metadata": {
                "model_load_strategy": "single_device",
                "offload_state_dict": True,
            },
            "test_python": "python",
            "test_python_source": "map_default",
        }
        for _ in range(2)
    ]


def test_phase2bb_audit_accepts_clone_present_package_loaded_matrix(
    tmp_path: Path,
) -> None:
    report = audit_phase2bb_package_loaded_execution_matrix(
        subset_report_json=_write_json(tmp_path / "subset.json", _subset()),
        package_gate_json=_write_json(tmp_path / "gate.json", _package_gate()),
        package_execution_summary_json=_write_json(
            tmp_path / "execution.json", _execution()
        ),
        package_execution_jsonl=_write_jsonl(
            tmp_path / "execution.jsonl", _execution_rows()
        ),
        wrong_cache_summary_json=_write_json(tmp_path / "wrong.json", _wrong_cache()),
    )

    assert report["passed"] is True
    assert report["ready_for_phase2bb_clone_present_package_loaded_runtime_matrix"] is True
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert report["metrics"]["repo_count"] == 5
    assert report["metrics"]["clone_filtered_repos"] == ["filtered"]
    assert report["metrics"]["residual_rows"] == 2
    assert report["metrics"]["residual_failure_counts"] == {
        "missing_symbolic_structural_requirements": 2
    }


def test_phase2bb_audit_rejects_subset_without_clone_present_filter(
    tmp_path: Path,
) -> None:
    subset = _subset()
    subset["inputs"]["require_clone_present"] = False

    report = audit_phase2bb_package_loaded_execution_matrix(
        subset_report_json=_write_json(tmp_path / "subset.json", subset),
        package_gate_json=_write_json(tmp_path / "gate.json", _package_gate()),
        package_execution_summary_json=_write_json(
            tmp_path / "execution.json", _execution()
        ),
        wrong_cache_summary_json=_write_json(tmp_path / "wrong.json", _wrong_cache()),
    )

    assert report["passed"] is False
    assert report["checks"]["subset_clone_present_filter_enforced"] is False


def test_phase2bb_audit_advances_past_residual_repair_when_none_remain(
    tmp_path: Path,
) -> None:
    report = audit_phase2bb_package_loaded_execution_matrix(
        subset_report_json=_write_json(tmp_path / "subset.json", _subset()),
        package_gate_json=_write_json(tmp_path / "gate.json", _package_gate()),
        package_execution_summary_json=_write_json(
            tmp_path / "execution.json", _execution(success_rate=1.0)
        ),
        package_execution_jsonl=_write_jsonl(
            tmp_path / "execution.jsonl",
            [
                {
                    "success": True,
                    "phase2z_symbolic_patch_failure": None,
                    "stop_condition": "verification_passed",
                    "package_policy_metadata": {
                        "model_load_strategy": "single_device",
                        "offload_state_dict": True,
                    },
                    "test_python": "python",
                    "test_python_source": "map_default",
                }
                for _ in range(10)
            ],
        ),
        wrong_cache_summary_json=_write_json(tmp_path / "wrong.json", _wrong_cache()),
        min_execution_success_rate=1.0,
    )

    assert report["passed"] is True
    assert report["metrics"]["residual_rows"] == 0
    assert report["next_required_experiment"] == (
        "phase2bc_expand_clone_present_matrix_then_run_sealed_package_loaded_transfer"
    )


def test_phase2bb_audit_rejects_execution_rows_mismatching_subset(
    tmp_path: Path,
) -> None:
    report = audit_phase2bb_package_loaded_execution_matrix(
        subset_report_json=_write_json(tmp_path / "subset.json", _subset()),
        package_gate_json=_write_json(tmp_path / "gate.json", _package_gate()),
        package_execution_summary_json=_write_json(
            tmp_path / "execution.json", _execution(rows=8)
        ),
        wrong_cache_summary_json=_write_json(tmp_path / "wrong.json", _wrong_cache()),
    )

    assert report["passed"] is False
    assert report["checks"]["execution_rows_match_subset_rows"] is False
