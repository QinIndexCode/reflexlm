import json
from pathlib import Path

from reflexlm.cli.audit_phase2au_smoke_postflight import audit_phase2au_smoke_postflight


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(*, source_overlap: float = 0.4, candidates: int = 2, descriptor_count: int = 20) -> dict:
    return {
        "train_examples": 33,
        "val_examples": 20,
        "config_hash": "abc123",
        "source_overlap_command_slot_baseline": {
            "val": {"accuracy": source_overlap}
        },
        "pairwise_candidate_encoding": {
            "val": {"max_valid_candidates_per_row": candidates}
        },
        "history": [
            {
                "val_metrics": {
                    "command_slot_accuracy": 0.9,
                    "patch_operation_count": descriptor_count,
                    "patch_template_slot_count": descriptor_count,
                }
            }
        ],
    }


def test_phase2au_smoke_postflight_accepts_nontrivial_descriptor_smoke(tmp_path: Path) -> None:
    summary = _write(tmp_path / "summary.json", _summary())
    gate = _write(tmp_path / "pretrain.json", {"passed": True})

    report = audit_phase2au_smoke_postflight(
        training_summary_json=summary,
        pretrain_gate_json=gate,
    )

    assert report["passed"] is True
    assert report["ready_for_phase2au_full_training"] is True


def test_phase2au_smoke_postflight_rejects_source_overlap_ceiling(tmp_path: Path) -> None:
    summary = _write(tmp_path / "summary.json", _summary(source_overlap=1.0))
    gate = _write(tmp_path / "pretrain.json", {"passed": True})

    report = audit_phase2au_smoke_postflight(
        training_summary_json=summary,
        pretrain_gate_json=gate,
    )

    assert report["passed"] is False
    assert report["checks"]["source_overlap_not_ceiling"] is False
    assert "do_not_claim_learned_runtime_delta_from_this_smoke" in report["blocked_actions"]


def test_phase2au_smoke_postflight_rejects_missing_descriptor_counts(tmp_path: Path) -> None:
    summary = _write(tmp_path / "summary.json", _summary(descriptor_count=0))
    gate = _write(tmp_path / "pretrain.json", {"passed": True})

    report = audit_phase2au_smoke_postflight(
        training_summary_json=summary,
        pretrain_gate_json=gate,
    )

    assert report["passed"] is False
    assert report["checks"]["descriptor_operation_evaluable"] is False


def test_phase2au_smoke_postflight_rejects_single_candidate_command(tmp_path: Path) -> None:
    summary = _write(tmp_path / "summary.json", _summary(candidates=1))
    gate = _write(tmp_path / "pretrain.json", {"passed": True})

    report = audit_phase2au_smoke_postflight(
        training_summary_json=summary,
        pretrain_gate_json=gate,
    )

    assert report["passed"] is False
    assert report["checks"]["nontrivial_command_candidate_set"] is False
