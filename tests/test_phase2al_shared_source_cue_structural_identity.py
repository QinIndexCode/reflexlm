import json
from pathlib import Path

from reflexlm.cli.build_phase2al_shared_source_cue_structural_identity import (
    build_phase2al_shared_source_cue_structural_identity,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(index: int, *, expected_slot: int) -> dict:
    literal = f"from package_{index} import shared_symbol"
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


def test_phase2al_builds_non_ceiling_source_overlap_with_structural_identity(
    tmp_path: Path,
) -> None:
    rows = [_row(index, expected_slot=index % 3) for index in range(18)]
    source = _write_jsonl(tmp_path / "rows.jsonl", rows)

    report = build_phase2al_shared_source_cue_structural_identity(
        train_jsonl=source,
        val_jsonl=source,
        holdout_jsonl=source,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
        min_source_overlap_accuracy=0.20,
        max_source_overlap_accuracy=0.75,
    )

    assert report["passed"] is True
    assert report["claim_bearing_natural_trace_evidence"] is False
    assert report["split_metrics"]["val"]["identity_text_ablated_source_overlap"]["accuracy"] == 1 / 3
    assert report["split_metrics"]["val"]["runtime_identity_heuristic"]["accuracy"] == 1.0
    transformed = json.loads((tmp_path / "out" / "val.jsonl").read_text().splitlines()[0])
    assert transformed["phase2al_transform"]["runtime_structural_probe_hashes_preserved"]
    assert {
        candidate["target_symbol"] for candidate in transformed["repair_candidates"]
    } == {"from package_0 import shared_symbol"}


def test_phase2al_rejects_source_overlap_ceiling(tmp_path: Path) -> None:
    rows = [_row(index, expected_slot=0) for index in range(18)]
    source = _write_jsonl(tmp_path / "rows.jsonl", rows)

    report = build_phase2al_shared_source_cue_structural_identity(
        train_jsonl=source,
        val_jsonl=source,
        holdout_jsonl=source,
        output_dir=tmp_path / "out",
        min_source_overlap_accuracy=0.20,
        max_source_overlap_accuracy=0.75,
    )

    assert report["passed"] is False
    assert report["split_checks"]["val"]["source_overlap_not_ceiling"] is False
    assert "do_not_train_phase2al_until_shared_source_cue_gate_passes" in report[
        "blocked_actions"
    ]
