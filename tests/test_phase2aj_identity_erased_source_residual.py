import json
from pathlib import Path

from reflexlm.cli.build_phase2af_hardened_structural_sidecar_split import (
    _row_candidate,
)
from reflexlm.cli.build_phase2aj_identity_erased_source_residual import (
    build_phase2aj_identity_erased_source_residual,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(index: int, *, expected_slot: int = 1) -> dict:
    candidates = [
        {
            "repair_action": f"structural_repair_probe_{index}_{slot}",
            "intent": "apply_patch_and_rerun_tests",
            "edit_scope": "bounded_public_source_patch",
            "structural_probe_hash": f"structural_repair_probe_{index}_{slot}",
            "target_symbol": f"symbol_{index}_{slot}",
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
        "current_visible_text": (
            "public structural repair evidence "
            f"{candidates[expected_slot]['repair_action']}"
        ),
        "runtime_visible_evidence": {
            "changed_files": ["pkg/module.py"],
            "structural_probe_hashes": [candidates[expected_slot]["structural_probe_hash"]],
            "watched_files": ["phase2z_repair_tests/test_case.py"],
            "pytest_before_patch": {
                "stdout_excerpt": f"AssertionError near symbol_{index}_{expected_slot}"
            },
        },
        "repair_candidates": candidates,
        "expected_repair_action": candidates[expected_slot]["repair_action"],
        "expected_repair_result": {"test_target": "phase2z_repair_tests/test_case.py"},
        "normalization": {"sealed_feedback_absent": True},
    }


def test_phase2aj_adds_ambiguous_identity_receptor_and_preserves_source_residual(
    tmp_path: Path,
) -> None:
    rows = [_row(index, expected_slot=1 + index % 2) for index in range(18)]
    source = _write_jsonl(tmp_path / "rows.jsonl", rows)

    report = build_phase2aj_identity_erased_source_residual(
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
    assert "structural_probe_hash" in transformed["repair_candidates"][0]
    assert transformed["repair_candidates"][0]["identity_sidecar_ambiguity_control"] is True
    assert "command_identity_tokens=" in transformed["current_visible_text"]
    assert transformed["phase2aj_transform"]["candidate_source_metadata_preserved"] is True
    assert (
        transformed["phase2aj_transform"][
            "ambiguous_identity_receptor_contains_all_candidate_actions"
        ]
        is True
    )
    assert (
        transformed["phase2aj_measured_controls"][
            "controlled_ambiguous_identity_prediction"
        ]
        == 0
    )
    assert (
        transformed["phase2aj_measured_controls"][
            "controlled_ambiguous_identity_correct"
        ]
        is False
    )
    candidate = _row_candidate(transformed, require_tie_residual_feasible=True)
    assert candidate is not None
    assert (
        candidate["phase2af_measured_shortcuts"]["correct"][
            "identity_text_ablated_source_overlap"
        ]
        is True
    )


def test_phase2aj_rejects_insufficient_source_residual_rows(tmp_path: Path) -> None:
    source = _write_jsonl(tmp_path / "rows.jsonl", [_row(0, expected_slot=0)])

    report = build_phase2aj_identity_erased_source_residual(
        train_jsonl=source,
        val_jsonl=source,
        holdout_jsonl=source,
        output_dir=tmp_path / "out",
        min_train_rows=2,
        min_val_rows=2,
        min_holdout_rows=2,
    )

    assert report["passed"] is False
    assert report["checks"]["train_rows_min"] is False
    assert "do_not_train_phase2aj_until_controlled_pressure_split_passes" in report[
        "blocked_actions"
    ]
