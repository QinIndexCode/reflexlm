import hashlib
import json
from pathlib import Path

from reflexlm.cli.audit_phase2ae_structural_sidecar_budget_pressure_patch_candidates import (
    audit_phase2ae_structural_sidecar_budget_pressure_patch_candidates,
)
from reflexlm.cli.audit_phase2ae_structural_sidecar_provenance import (
    audit_phase2ae_structural_sidecar_provenance,
)
from reflexlm.cli.audit_phase2ae_learning_gap import audit_phase2ae_learning_gap
from reflexlm.cli.audit_phase2ae_slot_support import audit_phase2ae_slot_support
from reflexlm.cli.build_phase2ae_structural_sidecar_budget_pressure_patch_candidates import (
    CLAIM_BOUNDARY,
    build_phase2ae_structural_sidecar_budget_pressure_patch_candidates,
    phase2s_row_to_phase2ae,
    structural_sidecar_prediction,
)
from reflexlm.cli.build_phase2ae_structural_sidecar_comparison import (
    build_phase2ae_structural_sidecar_comparison,
)
from reflexlm.cli.build_phase2s_head_dataset import _candidate_commands, _command_identity_signal
from reflexlm.cli.run_phase2aa_bounded_patch_candidate_execution import (
    _identity_prioritized_command_slot,
    _identity_signal_controlled_row,
)
from reflexlm.cli.run_phase2z_public_structural_repair_execution import _state_for_public_policy
from reflexlm.llm.receptor_latent import runtime_command_identity_signal


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
            "target_line": 10 + slot,
            "target_col": slot,
            "target_literal_hash": f"literal-{slot}",
            "verification_command": "python -m pytest -q <generated_repair_test> --maxfail=1",
        }
        for slot in range(4)
    ]
    expected = candidates[expected_slot]
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
            "target_location": {
                "path": expected["edit_scope"],
                "line": expected["target_line"],
                "col": expected["target_col"],
            },
            "expected_literal_hash": expected["target_literal_hash"],
            "watched_files": ["tests/test_generated.py"],
            "pytest_before_patch": {"stdout_excerpt": "assert actual == expected"},
        },
        "repair_candidates": candidates,
        "expected_repair_action": expected["repair_action"],
        "expected_repair_result": {"test_target": "phase2s_repair_tests/test_case.py"},
        "artifact_paths": {"patch_diff": "artifacts/patch.diff"},
        "baselines": {
            "source_overlap": candidates[0]["repair_action"],
            "prompt_only": candidates[0]["repair_action"],
            "react": candidates[0]["repair_action"],
            "modern_coding_agent_loop": candidates[0]["repair_action"],
            "native_head_only_no_cache": candidates[0]["repair_action"],
            "continuation_only": candidates[0]["repair_action"],
        },
        "normalization": {"sealed_feedback_absent": True},
    }


def test_phase2ae_builder_keeps_residual_rows_solved_by_structural_sidecar(tmp_path: Path) -> None:
    rows = [
        _row(0, "train", "train_repo", expected_slot=2),
        _row(1, "train", "train_repo", expected_slot=3),
        _row(2, "train", "train_repo", expected_slot=1),
    ]
    source = _write_jsonl(tmp_path / "train.raw.jsonl", rows)

    manifest = build_phase2ae_structural_sidecar_budget_pressure_patch_candidates(
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
    assert all(row["stripped_identity_heuristic_correct"] is False for row in built_rows)
    assert all(row["structural_sidecar_correct"] is True for row in built_rows)
    assert all(row["legacy_identity_neutralized_for_structural_sidecar"] is True for row in built_rows)
    assert all(
        candidate["edit_scope"] == "bounded_public_source_patch" and candidate["target_symbol"] == ""
        for row in built_rows
        for candidate in row["repair_candidates"]
    )
    assert all(structural_sidecar_prediction(row) == row["expected_patch_candidate_slot"] for row in built_rows)
    for row in built_rows:
        signal = _command_identity_signal(row, _candidate_commands(row))
        expected_slot = row["expected_patch_candidate_slot"]
        assert signal[f"command_identity_slot:{expected_slot}"] > 0.0
        assert signal["command_identity_margin"] > 0.0


def test_phase2ae_runtime_state_preserves_structural_sidecar_identity_tokens() -> None:
    row = phase2s_row_to_phase2ae(_row(0, "holdout", "repo", expected_slot=3))
    state = _state_for_public_policy(
        row=row,
        pre_test={
            "exit_code": 1,
            "duration_seconds": 0.1,
            "stdout": "assert actual == expected",
            "stderr": "",
        },
        test_rel="phase2s_repair_tests/test_case.py",
    )

    signal = runtime_command_identity_signal(state)
    scores = [signal[f"command_identity_slot:{slot}"] for slot in range(4)]

    assert scores[3] > 0.0
    assert scores.index(max(scores)) == 3


def test_phase2ae_audit_accepts_structural_sidecar_rows(tmp_path: Path) -> None:
    train = [
        phase2s_row_to_phase2ae(_row(index, "train", "train_repo", expected_slot=2 + index % 2))
        for index in range(30)
    ]
    val = [
        phase2s_row_to_phase2ae(_row(index, "val", "val_repo", expected_slot=2 + index % 2))
        for index in range(16)
    ]
    holdout = [
        phase2s_row_to_phase2ae(_row(index, "holdout", "holdout_repo", expected_slot=2 + index % 2))
        for index in range(30)
    ]
    report = audit_phase2ae_structural_sidecar_budget_pressure_patch_candidates(
        train_jsonl=_write_jsonl(tmp_path / "train.jsonl", train),
        val_jsonl=_write_jsonl(tmp_path / "val.jsonl", val),
        holdout_jsonl=_write_jsonl(tmp_path / "holdout.jsonl", holdout),
    )

    assert report["passed"] is True
    assert report["checks"]["stripped_identity_not_solving"] is True
    assert report["checks"]["legacy_identity_neutralized_for_structural_sidecar"] is True
    assert report["checks"]["structural_sidecar_solves_selected_rows"] is True


def test_phase2ae_audit_rejects_marker_leak_and_broken_structural_sidecar(tmp_path: Path) -> None:
    train = [
        phase2s_row_to_phase2ae(_row(index, "train", "train_repo", expected_slot=2 + index % 2))
        for index in range(30)
    ]
    val = [
        phase2s_row_to_phase2ae(_row(index, "val", "val_repo", expected_slot=2 + index % 2))
        for index in range(16)
    ]
    holdout = [
        phase2s_row_to_phase2ae(_row(index, "holdout", "holdout_repo", expected_slot=2 + index % 2))
        for index in range(30)
    ]
    val[0]["current_visible_text"] = "gold candidate_0 leaked"
    holdout[0]["runtime_visible_evidence"]["target_location"]["line"] = 999
    holdout[0]["runtime_visible_evidence"]["expected_literal_hash"] = "wrong-literal"

    report = audit_phase2ae_structural_sidecar_budget_pressure_patch_candidates(
        train_jsonl=_write_jsonl(tmp_path / "train.jsonl", train),
        val_jsonl=_write_jsonl(tmp_path / "val.jsonl", val),
        holdout_jsonl=_write_jsonl(tmp_path / "holdout.jsonl", holdout),
    )

    assert report["passed"] is False
    assert report["checks"]["no_marker_leak_in_visible_text"] is False
    assert report["checks"]["structural_sidecar_solves_selected_rows"] is False


def test_phase2ae_comparison_records_boundary_and_runtime_fix_delta(tmp_path: Path) -> None:
    data_health = {
        "passed": True,
        "metrics": {
            "structural_sidecar_accuracy": {"holdout": 1.0},
            "stripped_identity_accuracy": {"holdout": 0.0},
        },
    }
    full = {"success_rate": 1.0}
    policyless = {"success_rate": 0.0}
    prior = {"success_rate": 0.42}
    erased = {"success_rate": 0.0}
    wrong = {"success_rate": 0.0}
    provenance = {"passed": True}
    (tmp_path / "data_health.json").write_text(json.dumps(data_health), encoding="utf-8")
    (tmp_path / "full.json").write_text(json.dumps(full), encoding="utf-8")
    (tmp_path / "policyless.json").write_text(json.dumps(policyless), encoding="utf-8")
    (tmp_path / "prior.json").write_text(json.dumps(prior), encoding="utf-8")
    (tmp_path / "erased.json").write_text(json.dumps(erased), encoding="utf-8")
    (tmp_path / "wrong.json").write_text(json.dumps(wrong), encoding="utf-8")
    (tmp_path / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    report = build_phase2ae_structural_sidecar_comparison(
        data_health_json=tmp_path / "data_health.json",
        provenance_audit_json=tmp_path / "provenance.json",
        full_summary_json=tmp_path / "full.json",
        policyless_summary_json=tmp_path / "policyless.json",
        prior_full_summary_json=tmp_path / "prior.json",
        erased_structural_summary_json=tmp_path / "erased.json",
        wrong_structural_summary_json=tmp_path / "wrong.json",
        output_json=tmp_path / "comparison.json",
    )

    assert report["passed"] is True
    assert report["checks"]["provenance_audit_passed"] is True
    assert report["metrics"]["runtime_signal_fix_delta"] == 0.5800000000000001
    assert report["metrics"]["full_minus_erased_structural"] == 1.0
    assert report["checks"]["wrong_structural_counterfactual_fails"] is True
    assert "learned_head_advantage_over_structural_sidecar" in report["unsupported_claims"]


def test_phase2ae_comparison_rejects_failed_provenance_audit(tmp_path: Path) -> None:
    data_health = {
        "passed": True,
        "metrics": {
            "structural_sidecar_accuracy": {"holdout": 1.0},
            "stripped_identity_accuracy": {"holdout": 0.0},
        },
    }
    (tmp_path / "data_health.json").write_text(json.dumps(data_health), encoding="utf-8")
    (tmp_path / "full.json").write_text(json.dumps({"success_rate": 1.0}), encoding="utf-8")
    (tmp_path / "policyless.json").write_text(json.dumps({"success_rate": 0.0}), encoding="utf-8")
    (tmp_path / "provenance.json").write_text(json.dumps({"passed": False}), encoding="utf-8")

    report = build_phase2ae_structural_sidecar_comparison(
        data_health_json=tmp_path / "data_health.json",
        provenance_audit_json=tmp_path / "provenance.json",
        full_summary_json=tmp_path / "full.json",
        policyless_summary_json=tmp_path / "policyless.json",
        output_json=tmp_path / "comparison.json",
    )

    assert report["passed"] is False
    assert report["checks"]["provenance_audit_passed"] is False


def test_phase2ae_identity_signal_controls_toggle_structural_sidecar() -> None:
    row = phase2s_row_to_phase2ae(_row(0, "holdout", "repo", expected_slot=3))

    normal = _identity_signal_controlled_row(row, "normal")
    erased = _identity_signal_controlled_row(row, "erase_structural")
    wrong = _identity_signal_controlled_row(row, "wrong_structural")

    assert _identity_prioritized_command_slot(normal) == 3
    assert _identity_prioritized_command_slot(erased) is None
    assert _identity_prioritized_command_slot(wrong) in {0, 1, 2}


def test_phase2ae_provenance_audit_requires_saved_generated_test(tmp_path: Path) -> None:
    row = phase2s_row_to_phase2ae(_row(0, "holdout", "repo", expected_slot=3))
    expected_hash = hashlib.sha256("'literal-3'".encode("utf-8")).hexdigest()[:16]
    row["runtime_visible_evidence"]["expected_literal_hash"] = expected_hash
    row["repair_candidates"][3]["target_literal_hash"] = expected_hash
    row["artifact_paths"]["generated_test"] = "artifacts/holdout/repo/row_00000/generated_test.py"
    generated_test = tmp_path / row["artifact_paths"]["generated_test"]
    generated_test.parent.mkdir(parents=True)
    generated_test.write_text(
        "\n".join(
            [
                "import ast",
                "from pathlib import Path",
                "REPO_ROOT = Path(__file__).resolve().parents[1]",
                "TARGET_REL_PATH = 'pkg/shared.py'",
                "TARGET_LINE = 13",
                "TARGET_COL = 3",
                "",
                "def _literal_at_target_position():",
                "    return None",
                "",
                "def test_phase2s_public_repair_literal_restored():",
                "    assert _literal_at_target_position() == 'literal-3'",
            ]
        ),
        encoding="utf-8",
    )
    path = _write_jsonl(tmp_path / "rows.jsonl", [row])
    report = audit_phase2ae_structural_sidecar_provenance(
        train_jsonl=path,
        val_jsonl=path,
        holdout_jsonl=path,
        dataset_root=tmp_path,
    )
    assert report["passed"] is True

    row["artifact_paths"].pop("generated_test")
    missing = _write_jsonl(tmp_path / "missing.jsonl", [row])
    rejected = audit_phase2ae_structural_sidecar_provenance(
        train_jsonl=missing,
        val_jsonl=missing,
        holdout_jsonl=missing,
        dataset_root=tmp_path,
    )
    assert rejected["passed"] is False
    assert rejected["checks"]["generated_test_artifact_present"] is False


def test_phase2ae_learning_gap_audit_blocks_learned_head_overclaim(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "expected_patch_candidate_slot": 2,
            "initial_selected_patch_candidate_slot": 0,
            "identity_retry_slot": 2,
            "selected_patch_candidate_slot": 2,
            "success": True,
        },
        {
            "expected_patch_candidate_slot": 3,
            "initial_selected_patch_candidate_slot": 0,
            "identity_retry_slot": 3,
            "selected_patch_candidate_slot": 3,
            "success": True,
        },
    ]
    results = _write_jsonl(tmp_path / "results.jsonl", rows)

    report = audit_phase2ae_learning_gap(
        results_jsonl=results,
        output_json=tmp_path / "learning_gap.json",
    )

    assert report["passed"] is True
    assert report["metrics"]["initial_policy_selection_accuracy"] == 0.0
    assert report["metrics"]["identity_retry_rescue_rate"] == 1.0
    assert "learned native-head initial candidate selection" in report["interpretation"]["unsupported"]


def test_phase2ae_slot_support_audit_rejects_unseen_eval_slots(
    tmp_path: Path,
) -> None:
    summary = {
        "slot_intent_distribution": {
            "train": {"command_slots": {"0": 10, "1": 10}},
            "val": {"command_slots": {"0": 5, "1": 5}},
        }
    }
    rows = [
        {"expected_patch_candidate_slot": 2},
        {"expected_patch_candidate_slot": 3},
    ]
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    eval_rows = _write_jsonl(tmp_path / "eval.jsonl", rows)

    report = audit_phase2ae_slot_support(
        training_summary_json=tmp_path / "summary.json",
        eval_rows_jsonl=eval_rows,
        output_json=tmp_path / "slot_support.json",
    )

    assert report["passed"] is False
    assert report["checks"]["train_covers_eval_slots"] is False
    assert report["metrics"]["eval_slots_missing_from_train"] == ["2", "3"]


def test_phase2ae_slot_support_audit_accepts_covered_eval_slots(
    tmp_path: Path,
) -> None:
    summary = {
        "slot_intent_distribution": {
            "train": {"command_slots": {"2": 10, "3": 10}},
            "val": {"command_slots": {"2": 5, "3": 5}},
        }
    }
    rows = [
        {"expected_patch_candidate_slot": 2},
        {"expected_patch_candidate_slot": 3},
    ]
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    eval_rows = _write_jsonl(tmp_path / "eval.jsonl", rows)

    report = audit_phase2ae_slot_support(
        training_summary_json=tmp_path / "summary.json",
        eval_rows_jsonl=eval_rows,
        output_json=tmp_path / "slot_support.json",
    )

    assert report["passed"] is True
