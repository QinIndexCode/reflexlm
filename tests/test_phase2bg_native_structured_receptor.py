import json
from pathlib import Path

from reflexlm.cli.audit_phase2bg_native_structured_receptor import (
    audit_phase2bg_native_structured_receptor,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(control: str, success: float, *, probes: int) -> dict:
    rows = 12
    attempts = int(rows * success)
    return {
        "rows": rows,
        "success_rate": success,
        "execution_attempts": attempts,
        "attempt_success_rate": 1.0 if attempts else 0.0,
        "package_runtime_evidence_channel": "structured_receptor",
        "package_runtime_evidence_control": control,
        "package_runtime_evidence_prompt_present_rows": 0,
        "package_structural_probe_receptor_rows": probes,
        "package_policy_loaded_rows": rows,
        "package_qwen_called_rows": rows,
        "package_nsi_reference_override_rows": 0,
        "freeform_patch_generation_rows": 0,
        "sealed_feedback_used_rows": 0,
    }


def test_phase2bg_audit_accepts_structured_receptor_causal_matrix(
    tmp_path: Path,
) -> None:
    report = audit_phase2bg_native_structured_receptor(
        phase2bf_audit_json=_write(tmp_path / "phase2bf.json", {"passed": True}),
        normal_summary_json=_write(
            tmp_path / "normal.json", _summary("normal", 1.0, probes=12)
        ),
        erased_summary_json=_write(
            tmp_path / "erased.json", _summary("erased", 0.5, probes=0)
        ),
        wrong_summary_json=_write(
            tmp_path / "wrong.json", _summary("wrong", 0.0, probes=12)
        ),
    )

    assert report["passed"] is True
    assert report["ready_for_bounded_native_structured_receptor_claim"] is True
    assert report["ready_for_open_ended_native_perception_claim"] is False
    assert report["metrics"]["normal_minus_erased_success_delta"] == 0.5


def test_phase2bg_audit_rejects_prompt_leakage(tmp_path: Path) -> None:
    normal = _summary("normal", 1.0, probes=12)
    normal["package_runtime_evidence_prompt_present_rows"] = 12
    report = audit_phase2bg_native_structured_receptor(
        phase2bf_audit_json=_write(tmp_path / "phase2bf.json", {"passed": True}),
        normal_summary_json=_write(tmp_path / "normal.json", normal),
        erased_summary_json=_write(
            tmp_path / "erased.json", _summary("erased", 0.5, probes=0)
        ),
        wrong_summary_json=_write(
            tmp_path / "wrong.json", _summary("wrong", 0.0, probes=12)
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["runtime_evidence_absent_from_all_prompts"] is False
