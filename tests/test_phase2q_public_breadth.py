import json
from pathlib import Path

from reflexlm.cli.audit_phase2q_public_breadth import build_phase2q_public_breadth_gate


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
                    "rejected_reasons": [],
                }
            )
    return {
        "collector_family": "phase2m_public_repo_readonly_trace_collector",
        "sealed_v3_used": False,
        "writes_to_collected_repos": False,
        "structured_watch_keys": True,
        "include_behavior_summary": False,
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


def _phase2p_summary() -> dict:
    return {
        "passed": True,
        "sealed_v3_used_for_training_sampling_or_tuning": False,
    }


def test_phase2q_public_breadth_gate_accepts_disjoint_public_breadth(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "manifest.json", _manifest())
    data_health = _write(tmp_path / "data_health.json", _data_health())
    phase2p = _write(tmp_path / "phase2p.json", _phase2p_summary())

    report = build_phase2q_public_breadth_gate(
        collector_manifest_json=manifest,
        data_health_json=data_health,
        phase2p_summary_json=phase2p,
    )

    assert report["passed"] is True
    assert report["allowed_next_action"] == "run_phase2q_nonsealed_smoke_only"
    assert report["rollup"]["repo_count"] == 8
    assert report["checks"]["all_split_repos_disjoint"] is True


def test_phase2q_public_breadth_gate_rejects_sealed_training_signal(tmp_path: Path) -> None:
    manifest_payload = _manifest()
    manifest_payload["sealed_v3_used"] = True
    manifest = _write(tmp_path / "manifest.json", manifest_payload)
    data_health_payload = _data_health()
    data_health_payload["checks"]["phase2m_no_sealed_reference_anywhere"] = False
    data_health = _write(tmp_path / "data_health.json", data_health_payload)
    phase2p_payload = _phase2p_summary()
    phase2p_payload["sealed_v3_used_for_training_sampling_or_tuning"] = True
    phase2p = _write(tmp_path / "phase2p.json", phase2p_payload)

    report = build_phase2q_public_breadth_gate(
        collector_manifest_json=manifest,
        data_health_json=data_health,
        phase2p_summary_json=phase2p,
    )

    assert report["passed"] is False
    assert "do_not_use_sealed_or_sealed_failure_feedback" in report["blocked_actions"]
    assert "do_not_use_phase2p_sealed_results_as_training_signal" in report["blocked_actions"]


def test_phase2q_public_breadth_gate_rejects_shortcut_trace_shape(tmp_path: Path) -> None:
    manifest_payload = _manifest()
    manifest_payload["structured_watch_keys"] = False
    manifest_payload["include_behavior_summary"] = True
    manifest_payload["splits"]["holdout"]["repo_ids"] = ["train_a"]
    manifest = _write(tmp_path / "manifest.json", manifest_payload)
    data_health_payload = _data_health()
    data_health_payload["checks"]["phase2m_all_required_baselines_val_below_threshold"] = False
    data_health = _write(tmp_path / "data_health.json", data_health_payload)

    report = build_phase2q_public_breadth_gate(
        collector_manifest_json=manifest,
        data_health_json=data_health,
    )

    assert report["passed"] is False
    assert "do_not_train_phase2q_without_structured_watch_keys" in report["blocked_actions"]
    assert "do_not_train_phase2q_with_behavior_summary_shortcut" in report["blocked_actions"]
    assert "do_not_train_phase2q_when_baseline_solves_val" in report["blocked_actions"]
