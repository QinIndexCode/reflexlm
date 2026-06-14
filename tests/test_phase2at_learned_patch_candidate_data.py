import json
from pathlib import Path

from reflexlm.cli.audit_phase2at_learned_patch_candidate_data import (
    CLAIM_BOUNDARY,
    SCHEMA_VERSION,
    audit_phase2at_learned_patch_candidate_data,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _target(index: int) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "target_source": "learned_bounded_patch_candidate_target",
        "target_path": f"pkg/module_{index % 3}.py",
        "operation": "replace_attribute" if index % 2 else "replace_symbol",
        "anchor": {
            "kind": "ast_node",
            "symbol_hash": f"symbol-{index % 5}",
            "line_bucket": index % 7,
        },
        "before_fragment_hash": f"before-{index}",
        "after_fragment_template_id": f"template-{index % 4}",
        "literal_or_symbol_payload": {"symbol": f"repair_symbol_{index % 6}"},
        "safety_constraints": {
            "max_changed_files": 1,
            "allowed_paths": [f"pkg/module_{index % 3}.py"],
        },
        "verification_command_slot": index % 2,
    }


def _row(index: int, split: str) -> dict:
    expected = f"repair_action_{index % 2}"
    return {
        "trace_id": f"{split}:repo{index % 2}:{index}",
        "split": split,
        "benchmark_family": "phase2at_learned_bounded_patch_candidate_generation",
        "claim_boundary": CLAIM_BOUNDARY,
        "source_kind": "public_repo",
        "repo_id": f"{split}_repo_{index % 3}",
        "repo_url_or_origin": f"https://example.invalid/{split}_repo_{index % 3}.git",
        "current_visible_text": "runtime traceback and watched-file relation without markers",
        "runtime_visible_evidence": {
            "changed_files": [f"pkg/module_{index % 3}.py"],
            "traceback_symbol_hash": f"symbol-{index % 5}",
        },
        "repair_candidates": [
            {"repair_action": "repair_action_0", "patch_source": "learned_candidate_slot"},
            {"repair_action": "repair_action_1", "patch_source": "learned_candidate_slot"},
        ],
        "expected_repair_action": expected,
        "learned_patch_candidate_target": _target(index),
        "baselines": {
            "source_overlap": expected if index % 4 == 0 else "repair_action_0",
            "prompt_only": "repair_action_0",
        },
        "sealed_feedback_used": False,
    }


def _splits(tmp_path: Path, mutate=None) -> tuple[Path, Path, Path]:
    paths = []
    for split in ("train", "val", "holdout"):
        rows = [_row(i, split) for i in range(24)]
        if mutate:
            mutate(split, rows)
        paths.append(_write_jsonl(tmp_path / f"{split}.jsonl", rows))
    return paths[0], paths[1], paths[2]


def test_phase2at_data_health_accepts_learned_bounded_targets(tmp_path: Path) -> None:
    train, val, holdout = _splits(tmp_path)

    report = audit_phase2at_learned_patch_candidate_data(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
    )

    assert report["passed"] is True
    assert report["checks"]["no_recorded_or_symbolic_generation_targets"] is True
    assert report["checks"]["repo_origin_disjoint_val_holdout"] is True
    assert (
        "phase2at_data_ready_for_learned_bounded_patch_candidate_training"
        in report["supported_claims"]
    )


def test_phase2at_data_health_rejects_phase2aa_recorded_candidate_relabel(
    tmp_path: Path,
) -> None:
    def mutate(split: str, rows: list[dict]) -> None:
        rows[0]["repair_candidates"][0]["patch_source"] = "recorded_correct_patch_artifact"

    train, val, holdout = _splits(tmp_path, mutate)

    report = audit_phase2at_learned_patch_candidate_data(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
    )

    assert report["passed"] is False
    assert report["checks"]["no_recorded_or_symbolic_generation_targets"] is False
    assert (
        report["metrics"]["target_failure_reasons"][
            "candidate_uses_forbidden_patch_source"
        ]
        == 3
    )


def test_phase2at_data_health_rejects_freeform_diff_target(tmp_path: Path) -> None:
    def mutate(split: str, rows: list[dict]) -> None:
        rows[1]["learned_patch_candidate_target"]["patch_diff"] = "--- a/x\n+++ b/x\n"

    train, val, holdout = _splits(tmp_path, mutate)

    report = audit_phase2at_learned_patch_candidate_data(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
    )

    assert report["passed"] is False
    assert report["checks"]["no_recorded_or_symbolic_generation_targets"] is False
    assert (
        report["metrics"]["target_failure_reasons"][
            "freeform_or_recorded_diff_target_present"
        ]
        == 3
    )


def test_phase2at_data_health_rejects_marker_leak_and_repo_overlap(
    tmp_path: Path,
) -> None:
    def mutate(split: str, rows: list[dict]) -> None:
        if split == "val":
            rows[0]["current_visible_text"] = "gold candidate_0 leaked"
        if split == "holdout":
            rows[0]["repo_url_or_origin"] = "https://example.invalid/val_repo_0.git"

    train, val, holdout = _splits(tmp_path, mutate)

    report = audit_phase2at_learned_patch_candidate_data(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
    )

    assert report["passed"] is False
    assert report["checks"]["no_visible_marker_leak"] is False
    assert report["checks"]["repo_origin_disjoint_val_holdout"] is False
