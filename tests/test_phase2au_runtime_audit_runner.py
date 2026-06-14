import json
from pathlib import Path

from reflexlm.cli.audit_phase2au_runtime_delta_gate import PHASE2AU_RUNTIME_BOUNDARY
from reflexlm.cli.run_phase2au_policy_required_runtime_audit import (
    run_phase2au_policy_required_runtime_audit,
)
from reflexlm.llm.native_nervous_package import write_native_nervous_package


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _task(index: int, expected: str) -> dict:
    return {
        "task_id": f"phase2au:holdout:{index:05d}",
        "benchmark_family": "phase2au_policy_required_runtime_delta",
        "candidate_policy_commands": [
            "phase2au_apply_candidate --repair-action repair_a structural_probe_hash=aaa target_symbol=a --verify pytest",
            "phase2au_apply_candidate --repair-action repair_b structural_probe_hash=bbb target_symbol=b --verify pytest",
        ],
        "expected_repair_action": expected,
        "evaluation_commands": ["python -m pytest -q generated.py"],
        "allowed_write_scope": ["pkg/mod.py"],
        "difficulty_axes": ["ambiguous_nonliteral_semantic"],
        "runtime_visible_contract": {
            "policy_required_runtime_delta": True,
            "no_sealed_feedback": True,
        },
        "runtime_visible_identity": {
            "command_identity_tokens": ["bbb" if expected == "repair_b" else "aaa"],
            "sealed_feedback_used": False,
        },
        "expected_policy": {
            "patch_proposal": 1,
            "patch_operation": "apply_patch_and_rerun_tests",
            "patch_template": "symbol_reference_restoration",
        },
    }


def test_phase2au_runtime_audit_no_policy_first_candidate_boundary(
    tmp_path: Path,
) -> None:
    package_dir = tmp_path / "pkg"
    write_native_nervous_package(
        package_dir,
        base_model_name="qwen",
        native_head_path="heads",
        low_level_checkpoint_path="low.pt",
        policy_label="phase2au_package",
    )
    tasks = _write_jsonl(
        tmp_path / "tasks.jsonl",
        [_task(0, "repair_a"), _task(1, "repair_b")],
    )

    report = run_phase2au_policy_required_runtime_audit(
        tasks_jsonl=tasks,
        package_path=package_dir,
        output_jsonl=tmp_path / "rows.jsonl",
        summary_json=tmp_path / "summary.json",
        max_rows=2,
        load_policy=False,
    )

    assert report["passed"] is False
    assert report["claim_boundary"] == PHASE2AU_RUNTIME_BOUNDARY
    assert report["checks"]["all_rows_policy_loaded"] is False
    assert report["metrics"]["row_count"] == 2
    assert report["metrics"]["success_rate"] == 0.5
    assert "do_not_claim_patch_execution_success_from_command_selection_only" in report[
        "blocked_actions"
    ]
