import json
from pathlib import Path

from reflexlm.cli.build_phase2af_hardened_failure_audit import (
    build_phase2af_hardened_failure_audit,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _gate(*, source: float, identity: float, passed: bool = False) -> dict:
    metric = lambda value: {"correct": int(value * 100), "total": 100, "accuracy": value}
    return {
        "passed": passed,
        "split_metrics": {
            "val": {
                "identity_text_ablated_source_overlap": metric(source),
                "runtime_identity_heuristic": metric(identity),
            },
            "holdout": {
                "identity_text_ablated_source_overlap": metric(source),
                "runtime_identity_heuristic": metric(identity),
            },
        },
        "blocked_actions": ["do_not_train_phase2af_full"] if not passed else [],
    }


def test_phase2af_failure_audit_blocks_when_all_gates_fail(tmp_path: Path) -> None:
    identity_ceiling = _write(tmp_path / "identity.json", _gate(source=0.0, identity=1.0))
    source_ceiling = _write(tmp_path / "source.json", _gate(source=0.95, identity=0.8))

    report = build_phase2af_hardened_failure_audit(
        gate_jsons=[identity_ceiling, source_ceiling],
        output_json=tmp_path / "audit.json",
    )

    assert report["training_allowed"] is False
    assert report["claim_upgrade_allowed"] is False
    assert "do_not_train_phase2af_full" in report["blocked_actions"]
    assert report["issue_counts"]["identity_sidecar_ceiling_with_zero_nonidentity_control"] == 1
    assert report["issue_counts"]["source_overlap_ceiling_control_too_easy"] == 1


def test_phase2af_failure_audit_allows_training_only_if_any_gate_passed(tmp_path: Path) -> None:
    failed = _write(tmp_path / "failed.json", _gate(source=0.0, identity=1.0))
    passed = _write(tmp_path / "passed.json", _gate(source=0.35, identity=0.6, passed=True))

    report = build_phase2af_hardened_failure_audit(
        gate_jsons=[failed, passed],
        output_json=tmp_path / "audit.json",
    )

    assert report["training_allowed"] is True
    assert report["blocked_actions"] == []
    assert report["claim_upgrade_allowed"] is False
