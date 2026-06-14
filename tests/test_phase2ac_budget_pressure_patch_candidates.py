import json
from pathlib import Path

from reflexlm.cli.audit_phase2ac_budget_pressure_patch_candidates import (
    audit_phase2ac_budget_pressure_patch_candidates,
)
from reflexlm.cli.build_phase2ac_budget_pressure_patch_candidates import (
    CLAIM_BOUNDARY,
    build_phase2ac_budget_pressure_patch_candidates,
    identity_heuristic_prediction,
    phase2s_row_to_phase2ac,
)
from reflexlm.cli.build_phase2ac_budget_pressure_comparison import (
    build_phase2ac_budget_pressure_comparison,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(
    index: int,
    split: str = "val",
    repo: str | None = None,
    *,
    expected_slot: int = 2,
    correct_symbol: str = "target_symbol",
) -> dict:
    candidates = [
        {
            "repair_action": f"repair_{index}_{slot}",
            "intent": "apply_patch_and_rerun_tests",
            "edit_scope": f"pkg/module_{slot}.py",
            "target_symbol": correct_symbol if slot == expected_slot else f"distractor_{slot}",
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
        "current_visible_text": "public runtime evidence without oracle markers",
        "runtime_visible_evidence": {
            "changed_files": [f"pkg/module_{expected_slot}.py"],
            "traceback_symbols": [correct_symbol],
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


def test_phase2ac_builder_keeps_budget_pressure_identity_solved_rows(tmp_path: Path) -> None:
    rows = [
        _row(0, "train", "train_repo", expected_slot=2),
        _row(1, "train", "train_repo", expected_slot=3),
        _row(2, "train", "train_repo", expected_slot=1),
        _row(3, "train", "train_repo", expected_slot=2, correct_symbol="missing_symbol"),
    ]
    rows[3]["runtime_visible_evidence"]["traceback_symbols"] = ["unrelated"]
    rows[3]["runtime_visible_evidence"]["changed_files"] = ["pkg/unrelated.py"]
    source = _write_jsonl(tmp_path / "train.raw.jsonl", rows)

    manifest = build_phase2ac_budget_pressure_patch_candidates(
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
    assert all(row["expected_slot_outside_policyless_budget"] is True for row in built_rows)
    assert all(row["policyless_slot0_budget_expected_success"] is False for row in built_rows)
    assert all(identity_heuristic_prediction(row) == row["expected_patch_candidate_slot"] for row in built_rows)


def test_phase2ac_builder_round_robins_repos_to_reduce_prefix_bias(tmp_path: Path) -> None:
    rows = [
        _row(index, "holdout", repo, expected_slot=2 + index % 2)
        for repo in ["repo_a", "repo_b", "repo_c"]
        for index in range(3)
    ]
    source = _write_jsonl(tmp_path / "holdout.raw.jsonl", rows)

    build_phase2ac_budget_pressure_patch_candidates(
        train_jsonl=source,
        val_jsonl=source,
        holdout_jsonl=source,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
        attempt_budget=2,
    )

    built_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "holdout.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["repo_id"] for row in built_rows[:3]] == ["repo_a", "repo_b", "repo_c"]


def test_phase2ac_audit_accepts_budget_pressure_rows(tmp_path: Path) -> None:
    train = [
        phase2s_row_to_phase2ac(_row(index, "train", "train_repo", expected_slot=2 + index % 2))
        for index in range(60)
    ]
    val = [
        phase2s_row_to_phase2ac(_row(index, "val", "val_repo", expected_slot=2 + index % 2))
        for index in range(36)
    ]
    holdout = [
        phase2s_row_to_phase2ac(_row(index, "holdout", "holdout_repo", expected_slot=2 + index % 2))
        for index in range(72)
    ]
    train_path = _write_jsonl(tmp_path / "train.jsonl", train)
    val_path = _write_jsonl(tmp_path / "val.jsonl", val)
    holdout_path = _write_jsonl(tmp_path / "holdout.jsonl", holdout)

    report = audit_phase2ac_budget_pressure_patch_candidates(
        train_jsonl=train_path,
        val_jsonl=val_path,
        holdout_jsonl=holdout_path,
    )

    assert report["passed"] is True
    assert report["checks"]["repo_origin_disjoint"] is True
    assert report["checks"]["expected_slot_outside_policyless_budget"] is True
    assert report["checks"]["identity_heuristic_solves_selected_rows"] is True
    assert report["metrics"]["policyless_slot0_budget_expected_success_rate"]["holdout"] == 0.0


def test_phase2ac_audit_rejects_marker_leak_and_budget_violation(tmp_path: Path) -> None:
    train = [
        phase2s_row_to_phase2ac(_row(index, "train", "train_repo", expected_slot=2 + index % 2))
        for index in range(60)
    ]
    val = [
        phase2s_row_to_phase2ac(_row(index, "val", "val_repo", expected_slot=2 + index % 2))
        for index in range(36)
    ]
    holdout = [
        phase2s_row_to_phase2ac(_row(index, "holdout", "holdout_repo", expected_slot=2 + index % 2))
        for index in range(72)
    ]
    val[0]["current_visible_text"] = "gold candidate_0 leaked"
    holdout[0]["expected_patch_candidate_slot"] = 0
    holdout[0]["expected_slot_outside_policyless_budget"] = False
    holdout[0]["policyless_slot0_budget_expected_success"] = True
    train_path = _write_jsonl(tmp_path / "train.jsonl", train)
    val_path = _write_jsonl(tmp_path / "val.jsonl", val)
    holdout_path = _write_jsonl(tmp_path / "holdout.jsonl", holdout)

    report = audit_phase2ac_budget_pressure_patch_candidates(
        train_jsonl=train_path,
        val_jsonl=val_path,
        holdout_jsonl=holdout_path,
    )

    assert report["passed"] is False
    assert report["checks"]["no_marker_leak_in_visible_text"] is False
    assert report["checks"]["expected_slot_outside_policyless_budget"] is False
    assert report["checks"]["policyless_slot0_budget_expected_to_fail"] is False


def test_phase2ac_comparison_requires_absolute_success_gate(tmp_path: Path) -> None:
    data_health = tmp_path / "data_health.json"
    data_health.write_text(
        json.dumps({"metrics": {"identity_heuristic_accuracy": {"holdout": 1.0}}}),
        encoding="utf-8",
    )
    full_summary = tmp_path / "full_summary.json"
    policyless_summary = tmp_path / "policyless_summary.json"
    full_summary.write_text(json.dumps({"success_rate": 0.58}), encoding="utf-8")
    policyless_summary.write_text(json.dumps({"success_rate": 0.0}), encoding="utf-8")
    full_results = _write_jsonl(
        tmp_path / "full.jsonl",
        [
            {"initial_selected_patch_candidate_slot": 2, "expected_patch_candidate_slot": 2},
            {
                "initial_selected_patch_candidate_slot": 0,
                "selected_patch_candidate_slot": 1,
                "expected_patch_candidate_slot": 2,
            },
        ],
    )
    policyless_results = _write_jsonl(
        tmp_path / "policyless.jsonl",
        [
            {"initial_selected_patch_candidate_slot": 0, "expected_patch_candidate_slot": 2},
            {"initial_selected_patch_candidate_slot": 0, "expected_patch_candidate_slot": 2},
        ],
    )

    report = build_phase2ac_budget_pressure_comparison(
        data_health_json=data_health,
        full_summary_json=full_summary,
        full_results_jsonl=full_results,
        policyless_summary_json=policyless_summary,
        policyless_results_jsonl=policyless_results,
        output_json=tmp_path / "comparison.json",
    )

    assert report["passed"] is False
    assert report["interpretation"]["budget_constrained_advantage_over_policyless_supported"] is True
    assert report["interpretation"]["phase2ac_passes_success_gate"] is False
    assert report["interpretation"]["learned_native_head_advantage_over_identity_heuristic_supported"] is False
