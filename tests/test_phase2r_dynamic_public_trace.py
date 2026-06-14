import json
from pathlib import Path

from reflexlm.cli.audit_phase2r_dynamic_public_trace import (
    build_phase2r_dynamic_public_trace_gate,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _manifest() -> dict:
    repos = []
    splits = {
        "train": ["train_a", "train_b", "train_c", "train_d"],
        "val": ["val_a", "val_b"],
        "holdout": ["holdout_a", "holdout_b"],
    }
    for split, repo_ids in splits.items():
        for repo_id in repo_ids:
            repos.append(
                {
                    "repo_id": repo_id,
                    "split": split,
                    "rows_requested": 24,
                    "rows_emitted": 24,
                    "dynamic_execution_rows": 24,
                    "rejected_reasons": [],
                    "source_repo_read_only_observed": True,
                }
            )
    return {
        "collector_family": "phase2r_public_repo_dynamic_execution_trace_collector",
        "trace_construction_mode": "dynamic_public_repo_pytest_execution_trace",
        "sealed_v3_used": False,
        "writes_to_collected_repos": False,
        "execution_sandbox_used": True,
        "source_repo_read_only_observed": True,
        "structured_watch_keys": True,
        "include_behavior_summary": False,
        "dynamic_execution_rows": 8 * 24,
        "splits": {
            split: {"rows": len(repo_ids) * 24, "repo_ids": repo_ids}
            for split, repo_ids in splits.items()
        },
        "repos": repos,
    }


def _data_health(*, passed: bool = True) -> dict:
    return {
        "passed": passed,
        "checks": {
            "phase2m_no_sealed_reference_anywhere": True,
            "phase2m_no_candidate_slot_marker_visible": True,
            "phase2m_baseline_metadata_measured": True,
            "phase2m_baselines_match_computed_predictions": True,
            "phase2m_source_overlap_val_below_threshold": True,
            "phase2m_native_head_only_val_below_threshold": True,
            "phase2m_all_required_baselines_val_below_threshold": True,
        },
        "rollups": {"val": {"rows": 48}},
    }


def _phase2q_summary() -> dict:
    return {
        "sealed_v3_used_for_training_or_tuning": False,
        "gates": {"sealed_v3_gate_passed": True},
    }


def test_phase2r_dynamic_trace_gate_accepts_dynamic_public_breadth(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "manifest.json", _manifest())
    data_health = _write(tmp_path / "data_health.json", _data_health())
    phase2q = _write(tmp_path / "phase2q.json", _phase2q_summary())

    report = build_phase2r_dynamic_public_trace_gate(
        collector_manifest_json=manifest,
        data_health_json=data_health,
        phase2q_summary_json=phase2q,
    )

    assert report["passed"] is True
    assert report["allowed_next_action"] == "run_phase2r_nonsealed_smoke_only"
    assert report["checks"]["dynamic_execution_rows_match_split_rows"] is True
    assert report["checks"]["source_repo_read_only_observed"] is True


def test_phase2r_dynamic_trace_gate_rejects_static_or_missing_execution(
    tmp_path: Path,
) -> None:
    manifest_payload = _manifest()
    manifest_payload["collector_family"] = "phase2m_public_repo_readonly_trace_collector"
    manifest_payload["trace_construction_mode"] = "read_only_static_public_repo_trace"
    manifest_payload["execution_sandbox_used"] = False
    manifest_payload["dynamic_execution_rows"] = 0
    manifest = _write(tmp_path / "manifest.json", manifest_payload)
    data_health = _write(tmp_path / "data_health.json", _data_health())

    report = build_phase2r_dynamic_public_trace_gate(
        collector_manifest_json=manifest,
        data_health_json=data_health,
    )

    assert report["passed"] is False
    assert "do_not_train_phase2r_from_static_or_readonly_only_traces" in report["blocked_actions"]
    assert "do_not_train_phase2r_without_execution_sandbox" in report["blocked_actions"]
    assert "do_not_train_phase2r_until_every_row_has_dynamic_execution" in report[
        "blocked_actions"
    ]


def test_phase2r_dynamic_trace_gate_rejects_mutation_sealed_and_baseline_solved(
    tmp_path: Path,
) -> None:
    manifest_payload = _manifest()
    manifest_payload["sealed_v3_used"] = True
    manifest_payload["writes_to_collected_repos"] = True
    manifest_payload["source_repo_read_only_observed"] = False
    manifest = _write(tmp_path / "manifest.json", manifest_payload)
    data_health_payload = _data_health()
    data_health_payload["checks"]["phase2m_no_sealed_reference_anywhere"] = False
    data_health_payload["checks"]["phase2m_all_required_baselines_val_below_threshold"] = False
    data_health = _write(tmp_path / "data_health.json", data_health_payload)
    phase2q_payload = _phase2q_summary()
    phase2q_payload["sealed_v3_used_for_training_or_tuning"] = True
    phase2q = _write(tmp_path / "phase2q.json", phase2q_payload)

    report = build_phase2r_dynamic_public_trace_gate(
        collector_manifest_json=manifest,
        data_health_json=data_health,
        phase2q_summary_json=phase2q,
    )

    assert report["passed"] is False
    assert "do_not_use_sealed_or_sealed_failure_feedback" in report["blocked_actions"]
    assert "do_not_train_phase2r_if_collection_mutates_source_repos" in report[
        "blocked_actions"
    ]
    assert "do_not_train_phase2r_when_baseline_solves_val" in report["blocked_actions"]
    assert "do_not_use_phase2q_sealed_results_as_training_signal" in report[
        "blocked_actions"
    ]
