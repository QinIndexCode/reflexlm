import json
from pathlib import Path

from reflexlm.cli.audit_phase2bh_verified_plasticity_recall import (
    audit_phase2bh_verified_plasticity_recall,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(control: str, success: float, *, memory_hits: int = 0) -> dict:
    rows = 12
    return {
        "rows": rows,
        "success_rate": success,
        "execution_attempts": int(rows * success),
        "package_runtime_evidence_control": control,
        "package_runtime_evidence_prompt_present_rows": 0,
        "package_structural_probe_receptor_rows": rows if control == "normal" else 0,
        "package_plasticity_feedback_accepted_rows": rows if control == "normal" else 0,
        "package_plasticity_memory_hit_rows": memory_hits,
        "package_plasticity_control": "normal",
        "package_policy_loaded_rows": rows,
        "package_qwen_called_rows": rows,
        "package_nsi_reference_override_rows": 0,
        "freeform_patch_generation_rows": 0,
        "sealed_feedback_used_rows": 0,
    }


def _memory() -> dict:
    return {
        "schema_version": "reflexlm.synaptic_plasticity.v1",
        "feedback_events": 12,
        "connections": {
            f"pattern-{index}": {
                f"command-{index}": {
                    "command": f"repair-{index}",
                    "weight": 1.0,
                    "verified_successes": 1,
                    "verified_failures": 0,
                }
            }
            for index in range(12)
        },
    }


def test_phase2bh_audit_accepts_verified_plasticity_recall(tmp_path: Path) -> None:
    wrong = _summary("identity_erased", 0.0, memory_hits=12)
    wrong["package_plasticity_control"] = "wrong"
    report = audit_phase2bh_verified_plasticity_recall(
        phase2bg_audit_json=_write(tmp_path / "bg.json", {"passed": True}),
        baseline_summary_json=_write(
            tmp_path / "baseline.json", _summary("identity_erased", 0.5)
        ),
        learning_summary_json=_write(
            tmp_path / "learning.json", _summary("normal", 1.0)
        ),
        recall_summary_json=_write(
            tmp_path / "recall.json", _summary("identity_erased", 1.0, memory_hits=12)
        ),
        wrong_memory_summary_json=_write(tmp_path / "wrong.json", wrong),
        memory_json=_write(tmp_path / "memory.json", _memory()),
    )

    assert report["passed"] is True
    assert report["ready_for_bounded_verifier_gated_plasticity_claim"] is True
    assert report["ready_for_differentiable_plasticity_claim"] is False


def test_phase2bh_audit_rejects_unverified_memory(tmp_path: Path) -> None:
    memory = _memory()
    memory["connections"]["pattern-0"]["command-0"]["verified_successes"] = 0
    wrong = _summary("identity_erased", 0.0, memory_hits=12)
    wrong["package_plasticity_control"] = "wrong"
    report = audit_phase2bh_verified_plasticity_recall(
        phase2bg_audit_json=_write(tmp_path / "bg.json", {"passed": True}),
        baseline_summary_json=_write(
            tmp_path / "baseline.json", _summary("identity_erased", 0.5)
        ),
        learning_summary_json=_write(
            tmp_path / "learning.json", _summary("normal", 1.0)
        ),
        recall_summary_json=_write(
            tmp_path / "recall.json", _summary("identity_erased", 1.0, memory_hits=12)
        ),
        wrong_memory_summary_json=_write(tmp_path / "wrong.json", wrong),
        memory_json=_write(tmp_path / "memory.json", memory),
    )

    assert report["passed"] is False
    assert report["checks"]["memory_connections_are_verified_success_only"] is False
