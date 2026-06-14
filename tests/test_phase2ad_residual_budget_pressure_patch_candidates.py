import json
from pathlib import Path

from reflexlm.cli.audit_phase2ad_residual_budget_pressure_patch_candidates import (
    audit_phase2ad_residual_budget_pressure_patch_candidates,
)
from reflexlm.cli.build_phase2ad_residual_budget_pressure_patch_candidates import (
    CLAIM_BOUNDARY,
    build_phase2ad_residual_budget_pressure_patch_candidates,
    phase2s_row_to_phase2ad,
)
from reflexlm.cli.build_phase2ad_residual_budget_pressure_comparison import (
    build_phase2ad_residual_budget_pressure_comparison,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(index: int, split: str = "val", repo: str | None = None, *, expected_slot: int = 2) -> dict:
    candidates = [
        {
            "repair_action": f"repair_{index}_{slot}",
            "intent": "apply_patch_and_rerun_tests",
            "edit_scope": "pkg/shared.py",
            "target_symbol": "shared_symbol",
            "verification_command": "python -m pytest -q <generated_repair_test> --maxfail=1",
        }
        for slot in range(4)
    ]
    return {
        "trace_id": f"{split}:{repo or split}:{index}",
        "split": split,
        "source_kind": "public_repo",
        "repo_id": repo or f"repo_{split}",
        "repo_url_or_origin": f"https://example.invalid/{repo or split}.git",
        "current_visible_text": "public runtime residual evidence without oracle markers",
        "runtime_visible_evidence": {
            "changed_files": ["pkg/shared.py"],
            "traceback_symbols": ["shared_symbol"],
            "watched_files": ["tests/test_generated.py"],
            "pytest_before_patch": {"stdout_excerpt": "assert actual == expected"},
        },
        "repair_candidates": candidates,
        "expected_repair_action": candidates[expected_slot]["repair_action"],
        "expected_repair_result": {"test_target": "phase2s_repair_tests/test_case.py"},
        "artifact_paths": {"patch_diff": "artifacts/patch.diff"},
        "baselines": {
            "source_overlap": candidates[expected_slot if index % 5 == 0 else 0]["repair_action"],
            "prompt_only": candidates[expected_slot if index % 4 == 0 else 0]["repair_action"],
            "react": candidates[expected_slot if index % 3 == 0 else 1]["repair_action"],
            "modern_coding_agent_loop": candidates[expected_slot if index % 2 == 0 else 1][
                "repair_action"
            ],
            "native_head_only_no_cache": candidates[0]["repair_action"],
            "continuation_only": candidates[1]["repair_action"],
        },
        "normalization": {"sealed_feedback_absent": True},
    }


def test_phase2ad_builder_keeps_identity_residual_budget_pressure_rows(tmp_path: Path) -> None:
    rows = [
        _row(0, "train", "train_repo", expected_slot=2),
        _row(1, "train", "train_repo", expected_slot=3),
        _row(2, "train", "train_repo", expected_slot=1),
    ]
    source = _write_jsonl(tmp_path / "train.raw.jsonl", rows)

    manifest = build_phase2ad_residual_budget_pressure_patch_candidates(
        train_jsonl=source,
        val_jsonl=source,
        holdout_jsonl=source,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
        attempt_budget=2,
    )

    built_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert manifest["split_counts"]["train"] == 2
    assert all(row["claim_boundary"] == CLAIM_BOUNDARY for row in built_rows)
    assert all(row["identity_heuristic_correct"] is False for row in built_rows)
    assert all(row["expected_slot_outside_policyless_budget"] is True for row in built_rows)
    assert all("first_outside_budget_slot" in row["baselines"] for row in built_rows)


def test_phase2ad_audit_accepts_residual_budget_pressure_rows(tmp_path: Path) -> None:
    train = [
        phase2s_row_to_phase2ad(_row(index, "train", "train_repo", expected_slot=2 + index % 2))
        for index in range(30)
    ]
    val = [
        phase2s_row_to_phase2ad(_row(index, "val", "val_repo", expected_slot=2 + index % 2))
        for index in range(16)
    ]
    holdout = [
        phase2s_row_to_phase2ad(_row(index, "holdout", "holdout_repo", expected_slot=2 + index % 2))
        for index in range(30)
    ]
    train_path = _write_jsonl(tmp_path / "train.jsonl", train)
    val_path = _write_jsonl(tmp_path / "val.jsonl", val)
    holdout_path = _write_jsonl(tmp_path / "holdout.jsonl", holdout)

    report = audit_phase2ad_residual_budget_pressure_patch_candidates(
        train_jsonl=train_path,
        val_jsonl=val_path,
        holdout_jsonl=holdout_path,
    )

    assert report["passed"] is True
    assert report["checks"]["identity_heuristic_not_solving_selected_rows"] is True
    assert report["checks"]["expected_slot_outside_policyless_budget"] is True
    assert report["metrics"]["best_non_full_baseline_accuracy"] > 0.0


def test_phase2ad_audit_rejects_identity_solved_and_marker_leak(tmp_path: Path) -> None:
    train = [
        phase2s_row_to_phase2ad(_row(index, "train", "train_repo", expected_slot=2 + index % 2))
        for index in range(30)
    ]
    val = [
        phase2s_row_to_phase2ad(_row(index, "val", "val_repo", expected_slot=2 + index % 2))
        for index in range(16)
    ]
    holdout = [
        phase2s_row_to_phase2ad(_row(index, "holdout", "holdout_repo", expected_slot=2 + index % 2))
        for index in range(30)
    ]
    val[0]["current_visible_text"] = "gold candidate_0 leaked"
    holdout[0]["identity_heuristic_correct"] = True
    train_path = _write_jsonl(tmp_path / "train.jsonl", train)
    val_path = _write_jsonl(tmp_path / "val.jsonl", val)
    holdout_path = _write_jsonl(tmp_path / "holdout.jsonl", holdout)

    report = audit_phase2ad_residual_budget_pressure_patch_candidates(
        train_jsonl=train_path,
        val_jsonl=val_path,
        holdout_jsonl=holdout_path,
    )

    assert report["passed"] is False
    assert report["checks"]["no_marker_leak_in_visible_text"] is False
    assert report["checks"]["identity_heuristic_not_solving_selected_rows"] is False


def test_phase2ad_comparison_freezes_selector_insufficiency(tmp_path: Path) -> None:
    data_health = tmp_path / "data_health.json"
    data_health.write_text(
        json.dumps(
            {
                "metrics": {
                    "best_non_full_baseline_accuracy": 0.68,
                    "identity_heuristic_accuracy": {"holdout": 0.0},
                }
            }
        ),
        encoding="utf-8",
    )
    full_summary = tmp_path / "full_summary.json"
    policyless_summary = tmp_path / "policyless_summary.json"
    full_summary.write_text(json.dumps({"success_rate": 0.0}), encoding="utf-8")
    policyless_summary.write_text(json.dumps({"success_rate": 0.0}), encoding="utf-8")

    report = build_phase2ad_residual_budget_pressure_comparison(
        data_health_json=data_health,
        full_summary_json=full_summary,
        policyless_summary_json=policyless_summary,
        output_json=tmp_path / "comparison.json",
    )

    assert report["passed"] is False
    assert report["interpretation"]["selector_insufficiency_observed"] is True
    assert report["metrics"]["full_minus_best_non_full_baseline"] < 0.0
