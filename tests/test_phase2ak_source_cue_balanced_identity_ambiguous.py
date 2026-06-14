import json
from pathlib import Path

from reflexlm.cli.build_phase2af_hardened_structural_sidecar_split import (
    _row_candidate,
    _shortcut_key,
)
from reflexlm.cli.build_phase2ak_source_cue_balanced_identity_ambiguous import (
    build_phase2ak_source_cue_balanced_identity_ambiguous,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(index: int, *, expected_slot: int = 0) -> dict:
    literal = f"from package_{index} import target_{expected_slot}"
    candidates = [
        {
            "repair_action": f"structural_repair_probe_{index}_{slot}",
            "intent": "apply_patch_and_rerun_tests",
            "edit_scope": "bounded_public_source_patch",
            "structural_probe_hash": f"structural_probe_hash_{index}_{slot}",
            "target_symbol": f"distractor_symbol_{index}_{slot}",
            "verification_command": "python -m pytest -q <generated_repair_test> --maxfail=1",
        }
        for slot in range(3)
    ]
    return {
        "trace_id": f"train:repo:{index}",
        "split": "train",
        "source_kind": "public_repo",
        "repo_id": "repo",
        "repo_url_or_origin": "https://example.invalid/repo.git",
        "current_visible_text": "public structural repair evidence",
        "runtime_visible_evidence": {
            "changed_files": ["pkg/module.py"],
            "structural_probe_hashes": [candidates[expected_slot]["structural_probe_hash"]],
            "watched_files": ["phase2z_repair_tests/test_case.py"],
            "pytest_before_patch": {
                "stdout_excerpt": (
                    "F\n>       assert "
                    f"{literal!r} in text\nE       assert ...\n"
                )
            },
        },
        "repair_candidates": candidates,
        "expected_repair_action": candidates[expected_slot]["repair_action"],
        "expected_repair_result": {"test_target": "phase2z_repair_tests/test_case.py"},
        "normalization": {"sealed_feedback_absent": True},
    }


def test_phase2ak_builds_source_cue_balanced_identity_ambiguous_split(tmp_path: Path) -> None:
    rows = [_row(index, expected_slot=index % 3) for index in range(18)]
    source = _write_jsonl(tmp_path / "rows.jsonl", rows)

    report = build_phase2ak_source_cue_balanced_identity_ambiguous(
        train_jsonl=source,
        val_jsonl=source,
        holdout_jsonl=source,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
    )

    assert report["passed"] is True
    assert report["claim_bearing_natural_trace_evidence"] is False
    assert report["bucket_counts"]["holdout"] == {"source_1_identity_0": 18}
    transformed = json.loads((tmp_path / "out" / "holdout.jsonl").read_text().splitlines()[0])
    assert transformed["phase2ak_transform"]["source_cue_from_runtime_visible_assert_literal"]
    assert transformed["phase2ak_transform"]["uses_sealed_feedback"] is False
    candidate = _row_candidate(transformed, require_tie_residual_feasible=False)
    assert candidate is not None
    assert _shortcut_key(candidate) == (1, 0)


def test_phase2ak_rejects_rows_without_runtime_visible_assert_literal(tmp_path: Path) -> None:
    row = _row(0)
    row["runtime_visible_evidence"]["pytest_before_patch"]["stdout_excerpt"] = "AssertionError"
    source = _write_jsonl(tmp_path / "rows.jsonl", [row])

    report = build_phase2ak_source_cue_balanced_identity_ambiguous(
        train_jsonl=source,
        val_jsonl=source,
        holdout_jsonl=source,
        output_dir=tmp_path / "out",
        min_train_rows=1,
        min_val_rows=1,
        min_holdout_rows=1,
    )

    assert report["passed"] is False
    assert report["split_counts"]["train"] == 0
    assert "do_not_train_phase2ak_until_source_cue_balanced_split_passes" in report[
        "blocked_actions"
    ]
