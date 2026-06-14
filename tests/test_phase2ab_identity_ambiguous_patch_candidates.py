import json
from pathlib import Path

from reflexlm.cli.audit_phase2ab_identity_ambiguous_patch_candidates import (
    audit_phase2ab_identity_ambiguous_patch_candidates,
)
from reflexlm.cli.audit_phase2ab_identity_ambiguous_execution import (
    audit_phase2ab_identity_ambiguous_execution,
)
from reflexlm.cli.build_phase2ab_identity_ambiguous_patch_candidates import (
    CLAIM_BOUNDARY,
    build_phase2ab_identity_ambiguous_patch_candidates,
    phase2s_row_to_phase2ab,
)
from reflexlm.cli.build_phase2ab_retry_baseline_comparison import (
    build_phase2ab_retry_baseline_comparison,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(index: int, split: str = "val", repo: str | None = None) -> dict:
    expected_slot = index % 3
    candidates = [
        {
            "repair_action": f"repair_{index}_{slot}",
            "intent": "apply_patch_and_rerun_tests",
            "edit_scope": "pkg/module.py",
            "target_symbol": "same_symbol" if slot < 2 else "other_symbol",
            "verification_command": "python -m pytest -q <generated_repair_test> --maxfail=1",
        }
        for slot in range(3)
    ]
    return {
        "trace_id": f"{split}:{repo or split}:{index}",
        "split": split,
        "source_kind": "public_repo",
        "repo_id": repo or f"repo_{split}",
        "repo_url_or_origin": f"https://example.invalid/{repo or split}.git",
        "current_visible_text": "public runtime evidence without slot markers",
        "runtime_visible_evidence": {
            "changed_files": ["pkg/module.py"],
            "traceback_symbols": ["same_symbol"],
            "watched_files": ["tests/test_generated.py"],
            "pytest_before_patch": {"stdout_excerpt": "assert actual == expected"},
        },
        "repair_candidates": candidates,
        "expected_repair_action": candidates[expected_slot]["repair_action"],
        "expected_repair_result": {"test_target": "phase2s_repair_tests/test_case.py"},
        "artifact_paths": {"patch_diff": "artifacts/patch.diff"},
        "baselines": {
            "source_overlap": candidates[0]["repair_action"],
            "prompt_only": candidates[0]["repair_action"],
            "react": candidates[0]["repair_action"],
            "modern_coding_agent_loop": candidates[1]["repair_action"],
            "native_head_only_no_cache": candidates[0]["repair_action"],
            "continuation_only": candidates[0]["repair_action"],
        },
        "normalization": {"sealed_feedback_absent": True},
    }


def test_phase2ab_builder_keeps_only_identity_ambiguous_residual_rows(tmp_path: Path) -> None:
    rows = [
        _row(0, "train", "train_repo"),
        _row(1, "train", "train_repo"),
        {
            **_row(2, "train", "train_repo"),
            "runtime_visible_evidence": {
                **_row(2, "train", "train_repo")["runtime_visible_evidence"],
                "traceback_symbols": ["other_symbol"],
            },
        },
    ]
    source = _write_jsonl(tmp_path / "train.raw.jsonl", rows)

    manifest = build_phase2ab_identity_ambiguous_patch_candidates(
        train_jsonl=source,
        val_jsonl=source,
        holdout_jsonl=source,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
    )

    built_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert manifest["split_counts"]["train"] == 2
    assert all(row["claim_boundary"] == CLAIM_BOUNDARY for row in built_rows)
    assert all(row["identity_heuristic_correct"] is False for row in built_rows)
    assert all(row["requires_bounded_verification_retry"] is True for row in built_rows)


def test_phase2ab_conversion_removes_hash_identity_shortcuts() -> None:
    row = _row(1)
    row["runtime_visible_evidence"]["structural_probe_hashes"] = ["hash-correct"]
    row["runtime_visible_evidence"]["expected_literal_hash"] = "literal"
    row["repair_candidates"][1]["structural_probe_hash"] = "hash-correct"
    row["repair_candidates"][1]["target_literal_hash"] = "literal"

    converted = phase2s_row_to_phase2ab(row)

    assert converted["claim_boundary"] == CLAIM_BOUNDARY
    assert "structural_probe_hashes" not in converted["runtime_visible_evidence"]
    assert "expected_literal_hash" not in converted["runtime_visible_evidence"]
    assert all("structural_probe_hash" not in candidate for candidate in converted["repair_candidates"])
    assert all("target_literal_hash" not in candidate for candidate in converted["repair_candidates"])


def test_phase2ab_audit_accepts_identity_ambiguous_nonzero_controls(tmp_path: Path) -> None:
    train = [phase2s_row_to_phase2ab(_row(index, "train", "train_repo")) for index in range(60)]
    val = [phase2s_row_to_phase2ab(_row(index, "val", "val_repo")) for index in range(36)]
    holdout = [
        phase2s_row_to_phase2ab(_row(index, "holdout", "holdout_repo"))
        for index in range(72)
    ]
    train_path = _write_jsonl(tmp_path / "train.jsonl", train)
    val_path = _write_jsonl(tmp_path / "val.jsonl", val)
    holdout_path = _write_jsonl(tmp_path / "holdout.jsonl", holdout)

    report = audit_phase2ab_identity_ambiguous_patch_candidates(
        train_jsonl=train_path,
        val_jsonl=val_path,
        holdout_jsonl=holdout_path,
    )

    assert report["passed"] is True
    assert report["checks"]["repo_origin_disjoint"] is True
    assert report["checks"]["identity_heuristic_not_solving_val_or_holdout"] is True
    assert report["metrics"]["best_non_full_baseline_accuracy"] > 0.0


def test_phase2ab_audit_rejects_marker_leak_and_identity_shortcut(tmp_path: Path) -> None:
    train = [phase2s_row_to_phase2ab(_row(index, "train", "train_repo")) for index in range(60)]
    val = [phase2s_row_to_phase2ab(_row(index, "val", "val_repo")) for index in range(36)]
    holdout = [
        phase2s_row_to_phase2ab(_row(index, "holdout", "holdout_repo"))
        for index in range(72)
    ]
    val[0]["current_visible_text"] = "gold candidate_0 leaked"
    holdout[0]["repair_candidates"][0]["structural_probe_hash"] = "leak"
    train_path = _write_jsonl(tmp_path / "train.jsonl", train)
    val_path = _write_jsonl(tmp_path / "val.jsonl", val)
    holdout_path = _write_jsonl(tmp_path / "holdout.jsonl", holdout)

    report = audit_phase2ab_identity_ambiguous_patch_candidates(
        train_jsonl=train_path,
        val_jsonl=val_path,
        holdout_jsonl=holdout_path,
    )

    assert report["passed"] is False
    assert report["checks"]["no_marker_leak_in_visible_text"] is False
    assert report["checks"]["identity_shortcuts_absent"] is False


def test_phase2ab_execution_audit_tracks_retry_recovery_not_single_shot(tmp_path: Path) -> None:
    rows = [
        {
            "success": True,
            "policy_loaded": True,
            "bounded_candidate_retry_enabled": True,
            "initial_selected_patch_candidate_slot": 0,
            "selected_patch_candidate_slot": 1,
            "expected_patch_candidate_slot": 1,
            "candidate_attempts": [
                {
                    "candidate_slot": 0,
                    "patch_source": "selected_bounded_distractor_patch_candidate",
                    "passed": False,
                },
                {
                    "candidate_slot": 1,
                    "patch_source": "selected_recorded_correct_patch_candidate",
                    "passed": True,
                },
            ],
            "source_kind": "public_repo",
            "claim_bearing_freeform_patch_evidence": False,
            "freeform_patch_generation": False,
            "sealed_feedback_used": False,
            "false_completion": False,
            "generated_test_used": True,
        },
        {
            "success": True,
            "policy_loaded": True,
            "bounded_candidate_retry_enabled": True,
            "initial_selected_patch_candidate_slot": 0,
            "selected_patch_candidate_slot": 0,
            "expected_patch_candidate_slot": 0,
            "candidate_attempts": [
                {
                    "candidate_slot": 0,
                    "patch_source": "selected_recorded_correct_patch_candidate",
                    "passed": True,
                }
            ],
            "source_kind": "public_repo",
            "claim_bearing_freeform_patch_evidence": False,
            "freeform_patch_generation": False,
            "sealed_feedback_used": False,
            "false_completion": False,
            "generated_test_used": True,
        },
    ]
    path = _write_jsonl(tmp_path / "results.jsonl", rows)
    report = audit_phase2ab_identity_ambiguous_execution(
        execution_results_jsonl=path,
        min_rows=2,
        min_success_rate=0.85,
        min_retry_recoveries=1,
    )

    assert report["passed"] is True
    assert report["metrics"]["initial_selection_accuracy"] == 0.5
    assert report["metrics"]["final_selection_accuracy_after_retry"] == 1.0
    assert report["metrics"]["retry_recovery_count"] == 1


def test_phase2ab_retry_baseline_comparison_blocks_unique_full_claim(tmp_path: Path) -> None:
    full_summary = tmp_path / "full_summary.json"
    policyless_summary = tmp_path / "policyless_summary.json"
    full_summary.write_text(json.dumps({"success_rate": 1.0}), encoding="utf-8")
    policyless_summary.write_text(json.dumps({"success_rate": 1.0}), encoding="utf-8")
    full_results = _write_jsonl(
        tmp_path / "full.jsonl",
        [
            {"initial_selected_patch_candidate_slot": 0, "expected_patch_candidate_slot": 1},
            {"initial_selected_patch_candidate_slot": 0, "expected_patch_candidate_slot": 0},
        ],
    )
    policyless_results = _write_jsonl(
        tmp_path / "policyless.jsonl",
        [
            {"initial_selected_patch_candidate_slot": 0, "expected_patch_candidate_slot": 1},
            {"initial_selected_patch_candidate_slot": 0, "expected_patch_candidate_slot": 0},
        ],
    )

    report = build_phase2ab_retry_baseline_comparison(
        full_summary_json=full_summary,
        full_results_jsonl=full_results,
        policyless_summary_json=policyless_summary,
        policyless_results_jsonl=policyless_results,
        output_json=tmp_path / "comparison.json",
    )

    assert report["metrics"]["full_minus_policyless_slot0_retry"] == 0.0
    assert report["interpretation"]["bounded_verification_loop_supported"] is True
    assert report["interpretation"]["full_package_unique_advantage_supported"] is False
