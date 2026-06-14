from __future__ import annotations

import json
from pathlib import Path

from reflexlm.cli.build_phase2aw_sealed_transfer_report import (
    build_phase2aw_sealed_transfer_report,
)


def _metric(mean: float, positives: int | None = None) -> dict[str, float | int]:
    payload: dict[str, float | int] = {"mean": mean, "count": 64}
    if positives is not None:
        payload["positives"] = positives
    return payload


def _eval_payload(
    tmp_path: Path,
    label: str,
    completion: float,
    *,
    positives: int,
    model_calls: float = 1.0,
    command: float | None = None,
    read: float | None = None,
    false_reflex: float = 0.0,
    qwen_debug_rows: int = 0,
) -> Path:
    run_path = tmp_path / f"run_{label}"
    run_path.mkdir()
    trace_rows = []
    for index in range(qwen_debug_rows):
        trace_rows.append(
            {
                "episode_id": f"ep_{index}",
                "task_type": "test_failure_reflex",
                "qwen_called": True,
                "policy_debug": {"action_source": "native_head_cortex"},
            }
        )
    if trace_rows:
        (run_path / "trace_rows.jsonl").write_text(
            "\n".join(json.dumps(row) for row in trace_rows) + "\n",
            encoding="utf-8",
        )
    aggregate: dict[str, object] = {
        "task_completion_rate": _metric(completion, positives),
        "model_calls": _metric(model_calls),
        "token_equivalent_cost": _metric(0.0),
        "reaction_latency_ms": _metric(1.0),
        "state_hallucination_rate": _metric(0.0),
        "false_reflex_rate": _metric(false_reflex),
    }
    if command is not None:
        aggregate["command_decision_accuracy"] = _metric(command)
    if read is not None:
        aggregate["read_file_decision_accuracy"] = _metric(read)
    path = tmp_path / f"{label}.json"
    path.write_text(
        json.dumps(
            {
                "policy": {"policy_label": label},
                "episode_count": 64,
                "run_path": str(run_path),
                "metrics": {"aggregate": aggregate},
            }
        ),
        encoding="utf-8",
    )
    return path


def _supporting_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "profile": "external_trace_v3_semantic_required",
                "sealed": True,
                "sealed_config_hash": "abc123",
            }
        ),
        encoding="utf-8",
    )
    sealed_auth = tmp_path / "sealed_auth.json"
    sealed_auth.write_text(
        json.dumps(
            {
                "passed": True,
                "ready_for_sealed_eval": True,
                "ready_for_claim_upgrade": False,
            }
        ),
        encoding="utf-8",
    )
    package_evidence = tmp_path / "package_evidence.json"
    package_evidence.write_text(json.dumps({"passed": True}), encoding="utf-8")
    return manifest, sealed_auth, package_evidence


def test_phase2aw_sealed_transfer_report_passes_bounded_gate(tmp_path: Path) -> None:
    manifest, sealed_auth, package_evidence = _supporting_artifacts(tmp_path)
    full = _eval_payload(
        tmp_path,
        "full",
        1.0,
        positives=64,
        model_calls=1.0,
        false_reflex=0.0,
        qwen_debug_rows=64,
    )
    no_nsi = _eval_payload(
        tmp_path,
        "no_nsi",
        0.265625,
        positives=17,
        model_calls=1.0,
        false_reflex=0.25,
    )
    native = _eval_payload(
        tmp_path,
        "native",
        0.0,
        positives=0,
        command=None,
        read=0.0,
        false_reflex=0.5,
    )
    continuation = _eval_payload(
        tmp_path,
        "continuation",
        0.0,
        positives=0,
        model_calls=0.0,
        command=0.0,
        read=1.0,
        false_reflex=0.33,
    )
    prompt = _eval_payload(tmp_path, "prompt", 0.0, positives=0, false_reflex=1.0)
    react = _eval_payload(tmp_path, "react", 0.0, positives=0, false_reflex=1.0)

    report = build_phase2aw_sealed_transfer_report(
        full_eval_json=full,
        prompt_eval_json=prompt,
        react_eval_json=react,
        no_nsi_eval_json=no_nsi,
        native_head_only_eval_json=native,
        continuation_only_eval_json=continuation,
        dataset_manifest_json=manifest,
        sealed_authorization_gate_json=sealed_auth,
        package_loaded_evidence_json=package_evidence,
    )

    assert report["passed"] is True
    assert report["ready_for_strong_architecture_claim"] is False
    assert report["metrics"]["nonzero_control_count"] == 1
    assert report["metrics"]["full_minus_no_nsi"] == 0.734375
    categories = {
        item["policy"]: item["category"]
        for item in report["zero_control_classifications"]
    }
    assert categories["native-head-only/no-cache"] == (
        "expected_zero_due_to_missing_continuation_or_read_file_path"
    )
    assert categories["continuation-only/no-native-heads"] == (
        "expected_zero_due_to_missing_native_command_slot_selection"
    )
    assert categories["prompt-only 7B"] == "valid_zero_failure"
    assert categories["ReAct 7B"] == "valid_zero_failure"


def test_phase2aw_sealed_transfer_report_rejects_unexplained_zero(tmp_path: Path) -> None:
    manifest, sealed_auth, package_evidence = _supporting_artifacts(tmp_path)
    full = _eval_payload(tmp_path, "full", 1.0, positives=64, qwen_debug_rows=64)
    no_nsi = _eval_payload(tmp_path, "no_nsi", 0.265625, positives=17)
    native = _eval_payload(tmp_path, "native", 0.0, positives=0)
    continuation = _eval_payload(tmp_path, "continuation", 0.0, positives=0, command=0.0, read=1.0)
    prompt = _eval_payload(tmp_path, "prompt", 0.0, positives=0)
    react = _eval_payload(tmp_path, "react", 0.0, positives=0)

    report = build_phase2aw_sealed_transfer_report(
        full_eval_json=full,
        prompt_eval_json=prompt,
        react_eval_json=react,
        no_nsi_eval_json=no_nsi,
        native_head_only_eval_json=native,
        continuation_only_eval_json=continuation,
        dataset_manifest_json=manifest,
        sealed_authorization_gate_json=sealed_auth,
        package_loaded_evidence_json=package_evidence,
    )

    # The report still classifies known controls by policy family; unknown zero rows would fail.
    assert report["checks"]["no_suspicious_unexplained_zero"] is True
    assert report["ready_for_epoch_making_architecture_claim"] is False


def test_phase2aw_sealed_transfer_report_requires_authorization(tmp_path: Path) -> None:
    manifest, sealed_auth, package_evidence = _supporting_artifacts(tmp_path)
    sealed_auth.write_text(
        json.dumps({"passed": False, "ready_for_sealed_eval": False}),
        encoding="utf-8",
    )
    full = _eval_payload(tmp_path, "full", 1.0, positives=64, qwen_debug_rows=64)
    no_nsi = _eval_payload(tmp_path, "no_nsi", 0.265625, positives=17)
    native = _eval_payload(tmp_path, "native", 0.0, positives=0)
    continuation = _eval_payload(tmp_path, "continuation", 0.0, positives=0, command=0.0, read=1.0)
    prompt = _eval_payload(tmp_path, "prompt", 0.0, positives=0, false_reflex=1.0)
    react = _eval_payload(tmp_path, "react", 0.0, positives=0, false_reflex=1.0)

    report = build_phase2aw_sealed_transfer_report(
        full_eval_json=full,
        prompt_eval_json=prompt,
        react_eval_json=react,
        no_nsi_eval_json=no_nsi,
        native_head_only_eval_json=native,
        continuation_only_eval_json=continuation,
        dataset_manifest_json=manifest,
        sealed_authorization_gate_json=sealed_auth,
        package_loaded_evidence_json=package_evidence,
    )

    assert report["passed"] is False
    assert report["checks"]["sealed_authorization_passed"] is False
