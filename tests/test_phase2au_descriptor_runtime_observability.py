import json
from pathlib import Path

from reflexlm.cli.audit_phase2au_descriptor_runtime_observability import (
    audit_phase2au_descriptor_runtime_observability,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _execution_row(index: int, operation: str, template: str) -> dict:
    return {
        "trace_id": f"holdout:repo:{index}",
        "policy_loaded": True,
        "success": True,
        "sealed_feedback_used": False,
        "freeform_patch_generation": False,
        "claim_bearing_freeform_patch_evidence": False,
        "policy_learned_patch_descriptor_outputs": {
            "patch_operation": operation,
            "patch_operation_index": index % 3,
            "patch_target_file_slot": 0,
            "patch_template": template,
            "patch_template_slot": index % 3,
        },
    }


def _source_row(index: int, mode: str) -> dict:
    return {
        "trace_id": f"holdout:repo:{index}",
        "runtime_visible_evidence": {"repair_modes": [mode]},
    }


def test_descriptor_runtime_observability_accepts_diverse_runtime_outputs(
    tmp_path: Path,
) -> None:
    execution_rows = [
        *[
            _execution_row(index, "insert_import", "import_restoration")
            for index in range(10)
        ],
        *[
            _execution_row(index, "replace_attribute", "call_attribute_restoration")
            for index in range(10, 20)
        ],
    ]
    source_rows = [
        *[_source_row(index, "behavioral_import_restoration") for index in range(10)],
        *[
            _source_row(index, "behavioral_attribute_restoration")
            for index in range(10, 20)
        ],
    ]

    report = audit_phase2au_descriptor_runtime_observability(
        execution_jsonl=_write_jsonl(tmp_path / "execution.jsonl", execution_rows),
        source_rows_jsonl=_write_jsonl(tmp_path / "source.jsonl", source_rows),
    )

    assert report["passed"] is True
    assert report["checks"]["descriptor_operation_template_diversity_met"] is True
    assert report["metrics"]["expected_match_rate"] == 1.0
    assert "phase2au_runtime_descriptor_outputs_present_and_diverse" in report[
        "supported_claims"
    ]


def test_descriptor_runtime_observability_rejects_single_template_trace(
    tmp_path: Path,
) -> None:
    execution_rows = [
        _execution_row(index, "insert_import", "import_restoration")
        for index in range(20)
    ]

    report = audit_phase2au_descriptor_runtime_observability(
        execution_jsonl=_write_jsonl(tmp_path / "execution.jsonl", execution_rows),
    )

    assert report["passed"] is False
    assert report["checks"]["descriptor_outputs_present"] is True
    assert report["checks"]["descriptor_operation_template_diversity_met"] is False
    assert "do_not_claim_descriptor_runtime_generation_from_single_template_trace" in report[
        "blocked_actions"
    ]


def test_descriptor_runtime_observability_rejects_missing_or_invalid_outputs(
    tmp_path: Path,
) -> None:
    rows = [
        _execution_row(index, "insert_import", "import_restoration")
        for index in range(20)
    ]
    rows[0]["policy_learned_patch_descriptor_outputs"] = {}
    rows[1]["policy_learned_patch_descriptor_outputs"]["patch_operation"] = "freeform_diff"

    report = audit_phase2au_descriptor_runtime_observability(
        execution_jsonl=_write_jsonl(tmp_path / "execution.jsonl", rows),
    )

    assert report["passed"] is False
    assert report["checks"]["descriptor_outputs_present"] is False
    assert report["checks"]["descriptor_outputs_allowlisted"] is False
