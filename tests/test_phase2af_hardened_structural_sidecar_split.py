import json
from pathlib import Path

from reflexlm.cli.build_phase2af_hardened_structural_sidecar_split import (
    _candidate_order_variants,
    _expected_slot_for_manifest,
    audit_phase2af_hardened_structural_sidecar_split,
    build_phase2af_hardened_structural_sidecar_split,
    build_phase2af_stratified_hardened_structural_sidecar_split,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(index: int, *, expected_slot: int, visible_symbol_slot: int, split: str = "val") -> dict:
    candidates = [
        {
            "repair_action": f"repair_action_{index}_{slot}",
            "intent": "apply_patch_and_rerun_tests",
            "edit_scope": "pkg/module.py",
            "target_symbol": f"symbol_{index}_{slot}",
            "verification_command": "python -m pytest -q <generated_repair_test> --maxfail=1",
        }
        for slot in range(4)
    ]
    return {
        "trace_id": f"{split}:repo_{index}:{index}",
        "split": split,
        "source_kind": "public_repo",
        "repo_id": f"repo_{index % 3}",
        "repo_url_or_origin": f"https://example.invalid/repo_{index % 3}.git",
        "current_visible_text": "public runtime repair evidence without oracle markers",
        "runtime_visible_evidence": {
            "changed_files": ["pkg/module.py"],
            "traceback_symbols": [f"symbol_{index}_{visible_symbol_slot}"],
            "target_location": {"path": "pkg/module.py"},
            "watched_files": ["tests/test_generated.py"],
            "pytest_before_patch": {"stdout_excerpt": "AssertionError"},
        },
        "repair_candidates": candidates,
        "expected_repair_action": candidates[expected_slot]["repair_action"],
        "expected_repair_result": {"test_target": "phase2s_repair_tests/test_case.py"},
        "normalization": {"sealed_feedback_absent": True},
    }


def test_phase2af_split_gate_accepts_nonzero_non_ceiling_shortcuts(tmp_path: Path) -> None:
    rows = [
        _row(0, expected_slot=1, visible_symbol_slot=1),
        _row(1, expected_slot=1, visible_symbol_slot=0),
        _row(2, expected_slot=2, visible_symbol_slot=0),
        _row(3, expected_slot=3, visible_symbol_slot=0),
    ]
    source = _write_jsonl(tmp_path / "rows.jsonl", rows)

    manifest = build_phase2af_hardened_structural_sidecar_split(
        train_jsonl=source,
        val_jsonl=source,
        holdout_jsonl=source,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
    )
    report = audit_phase2af_hardened_structural_sidecar_split(
        manifest_json=tmp_path / "manifest.json",
        min_val_rows=4,
        min_holdout_rows=4,
    )

    assert manifest["artifact_family"] == "phase2af_hardened_structural_sidecar_split"
    assert manifest["split_metrics"]["val"]["identity_text_ablated_source_overlap"]["accuracy"] == 0.25
    assert manifest["tie_residual_feasibility"]["val"]["unresolved_identity_tie_rows"] == 0
    assert report["passed"] is True


def test_phase2af_split_gate_rejects_unresolved_identity_ties(tmp_path: Path) -> None:
    rows = [
        _row(index, expected_slot=1, visible_symbol_slot=1)
        for index in range(4)
    ]
    for row in rows:
        for candidate in row["repair_candidates"]:
            candidate["target_symbol"] = "same_symbol"
        row["runtime_visible_evidence"]["traceback_symbols"] = ["same_symbol"]
    source = _write_jsonl(tmp_path / "rows.jsonl", rows)

    manifest = build_phase2af_hardened_structural_sidecar_split(
        train_jsonl=source,
        val_jsonl=source,
        holdout_jsonl=source,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
    )
    report = audit_phase2af_hardened_structural_sidecar_split(
        manifest_json=tmp_path / "manifest.json",
        min_val_rows=4,
        min_holdout_rows=4,
    )

    assert manifest["tie_residual_feasibility"]["val"]["unresolved_identity_tie_rows"] == 4
    assert report["passed"] is False
    assert report["checks"]["val_no_unresolved_identity_ties"] is False


def test_phase2af_builder_can_filter_unresolved_identity_tie_rows(
    tmp_path: Path,
) -> None:
    good_rows = [
        _row(index, expected_slot=1, visible_symbol_slot=1)
        for index in range(4)
    ]
    tied_rows = [
        _row(100 + index, expected_slot=1, visible_symbol_slot=1)
        for index in range(4)
    ]
    for row in tied_rows:
        for candidate in row["repair_candidates"]:
            candidate["target_symbol"] = "same_symbol"
        row["runtime_visible_evidence"]["traceback_symbols"] = ["same_symbol"]
    source = _write_jsonl(tmp_path / "rows.jsonl", good_rows + tied_rows)

    manifest = build_phase2af_hardened_structural_sidecar_split(
        train_jsonl=source,
        val_jsonl=source,
        holdout_jsonl=source,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
        require_tie_residual_feasible_rows=True,
    )
    report = audit_phase2af_hardened_structural_sidecar_split(
        manifest_json=tmp_path / "manifest.json",
        min_val_rows=4,
        min_holdout_rows=4,
        max_raw_source_accuracy=1.0,
        max_text_ablated_source_accuracy=1.0,
        max_runtime_identity_accuracy=1.0,
    )

    assert manifest["require_tie_residual_feasible_rows"] is True
    assert manifest["split_counts"]["val"] == 4
    assert manifest["tie_residual_feasibility"]["val"]["unresolved_identity_tie_rows"] == 0
    assert report["checks"]["val_no_unresolved_identity_ties"] is True


def test_phase2af_split_gate_rejects_legacy_manifest_without_tie_residual(
    tmp_path: Path,
) -> None:
    rows = [
        _row(0, expected_slot=1, visible_symbol_slot=1),
        _row(1, expected_slot=1, visible_symbol_slot=0),
        _row(2, expected_slot=2, visible_symbol_slot=0),
        _row(3, expected_slot=3, visible_symbol_slot=0),
    ]
    source = _write_jsonl(tmp_path / "rows.jsonl", rows)
    manifest = build_phase2af_hardened_structural_sidecar_split(
        train_jsonl=source,
        val_jsonl=source,
        holdout_jsonl=source,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
    )
    manifest.pop("tie_residual_feasibility")
    (tmp_path / "legacy_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_phase2af_hardened_structural_sidecar_split(
        manifest_json=tmp_path / "legacy_manifest.json",
        min_val_rows=4,
        min_holdout_rows=4,
    )

    assert report["passed"] is False
    assert report["checks"]["tie_residual_feasibility_present"] is False


def test_phase2af_split_gate_rejects_shortcut_ceiling(tmp_path: Path) -> None:
    rows = [_row(index, expected_slot=1, visible_symbol_slot=1) for index in range(4)]
    source = _write_jsonl(tmp_path / "rows.jsonl", rows)

    build_phase2af_hardened_structural_sidecar_split(
        train_jsonl=source,
        val_jsonl=source,
        holdout_jsonl=source,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
    )
    report = audit_phase2af_hardened_structural_sidecar_split(
        manifest_json=tmp_path / "manifest.json",
        min_val_rows=4,
        min_holdout_rows=4,
    )

    assert report["passed"] is False
    assert report["checks"]["val_raw_source_not_ceiling"] is False
    assert "do_not_train_phase2af_full" in report["blocked_actions"]


def test_phase2af_builder_rejects_visible_candidate_or_gold_markers(tmp_path: Path) -> None:
    rows = [_row(index, expected_slot=1, visible_symbol_slot=1) for index in range(4)]
    rows[0]["current_visible_text"] = "gold_slot candidate_0 leakage"
    source = _write_jsonl(tmp_path / "rows.jsonl", rows)

    manifest = build_phase2af_hardened_structural_sidecar_split(
        train_jsonl=source,
        val_jsonl=source,
        holdout_jsonl=source,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
    )

    assert manifest["eligible_split_counts"]["val"] == 3
    built_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "val.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all("phase2af_measured_shortcuts" in row for row in built_rows)


def test_phase2af_split_gate_can_require_repo_disjoint_splits(tmp_path: Path) -> None:
    shared_rows = [
        _row(0, expected_slot=1, visible_symbol_slot=1),
        _row(1, expected_slot=1, visible_symbol_slot=0),
        _row(2, expected_slot=2, visible_symbol_slot=0),
        _row(3, expected_slot=3, visible_symbol_slot=0),
    ]
    shared = _write_jsonl(tmp_path / "shared.jsonl", shared_rows)
    build_phase2af_hardened_structural_sidecar_split(
        train_jsonl=shared,
        val_jsonl=shared,
        holdout_jsonl=shared,
        output_dir=tmp_path / "shared_out",
        manifest_json=tmp_path / "shared_manifest.json",
    )
    rejected = audit_phase2af_hardened_structural_sidecar_split(
        manifest_json=tmp_path / "shared_manifest.json",
        min_val_rows=4,
        min_holdout_rows=4,
        require_repo_disjoint=True,
    )
    assert rejected["passed"] is False
    assert rejected["checks"]["repo_origin_disjoint"] is False
    assert rejected["repo_overlaps"]

    def disjoint_rows(split: str, offset: int) -> list[dict]:
        rows = [
            _row(offset + index, expected_slot=1, visible_symbol_slot=1 if index == 0 else 0, split=split)
            for index in range(4)
        ]
        for row in rows:
            row["repo_id"] = f"{split}_{row['repo_id']}"
            row["repo_url_or_origin"] = f"https://example.invalid/{split}/{row['repo_id']}.git"
        return rows

    train = _write_jsonl(tmp_path / "train.jsonl", disjoint_rows("train", 0))
    val = _write_jsonl(tmp_path / "val.jsonl", disjoint_rows("val", 10))
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", disjoint_rows("holdout", 20))
    build_phase2af_hardened_structural_sidecar_split(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
        output_dir=tmp_path / "disjoint_out",
        manifest_json=tmp_path / "disjoint_manifest.json",
    )
    accepted = audit_phase2af_hardened_structural_sidecar_split(
        manifest_json=tmp_path / "disjoint_manifest.json",
        min_val_rows=4,
        min_holdout_rows=4,
        require_repo_disjoint=True,
    )
    assert accepted["passed"] is True
    assert accepted["repo_overlaps"] == {}


def test_phase2af_stratified_builder_composes_nonzero_non_ceiling_split(tmp_path: Path) -> None:
    source_wrong_identity_correct = [
        _row(index, expected_slot=1, visible_symbol_slot=0) for index in range(10)
    ]
    source_wrong_identity_wrong = [
        _row(100 + index, expected_slot=1, visible_symbol_slot=0) for index in range(4)
    ]
    for row in source_wrong_identity_wrong:
        candidates = row["repair_candidates"]
        row["expected_repair_action"] = candidates[2]["repair_action"]
    source_correct_identity_correct = [
        _row(200 + index, expected_slot=1, visible_symbol_slot=1) for index in range(10)
    ]

    identity_pressure = _write_jsonl(
        tmp_path / "identity_pressure.jsonl",
        source_wrong_identity_correct + source_wrong_identity_wrong,
    )
    source_feasible = _write_jsonl(tmp_path / "source_feasible.jsonl", source_correct_identity_correct)

    manifest = build_phase2af_stratified_hardened_structural_sidecar_split(
        train_jsonls=[identity_pressure, source_feasible],
        val_jsonls=[identity_pressure, source_feasible],
        holdout_jsonls=[identity_pressure, source_feasible],
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
        min_train_rows=12,
        min_val_rows=12,
        min_holdout_rows=12,
        max_train_rows=24,
        max_val_rows=24,
        max_holdout_rows=24,
    )
    report = audit_phase2af_hardened_structural_sidecar_split(
        manifest_json=tmp_path / "manifest.json",
        min_val_rows=12,
        min_holdout_rows=12,
    )

    assert manifest["build_mode"] == "multi_source_stratified_shortcut_controls"
    assert report["passed"] is True
    assert 0.05 <= manifest["split_metrics"]["val"]["identity_text_ablated_source_overlap"]["accuracy"] <= 0.75
    assert manifest["split_metrics"]["val"]["runtime_identity_heuristic"]["accuracy"] <= 0.90


def test_phase2af_stratified_builder_can_balance_train_shortcut_buckets(
    tmp_path: Path,
) -> None:
    source_wrong_identity_correct = [
        _row(index, expected_slot=1, visible_symbol_slot=0) for index in range(8)
    ]
    source_wrong_identity_wrong = [
        _row(100 + index, expected_slot=1, visible_symbol_slot=0) for index in range(3)
    ]
    for row in source_wrong_identity_wrong:
        candidates = row["repair_candidates"]
        row["expected_repair_action"] = candidates[2]["repair_action"]
    source_correct_identity_correct = [
        _row(200 + index, expected_slot=1, visible_symbol_slot=1) for index in range(12)
    ]
    source = _write_jsonl(
        tmp_path / "rows.jsonl",
        source_wrong_identity_correct + source_wrong_identity_wrong + source_correct_identity_correct,
    )

    manifest = build_phase2af_stratified_hardened_structural_sidecar_split(
        train_jsonls=[source],
        val_jsonls=[source],
        holdout_jsonls=[source],
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
        min_train_rows=12,
        min_val_rows=12,
        min_holdout_rows=12,
        max_train_rows=24,
        max_val_rows=24,
        max_holdout_rows=24,
        balance_train_shortcut_buckets=True,
        train_shortcut_bucket_target=5,
    )
    train_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert manifest["train_sampling"]["shortcut_bucket_balanced_train"] is True
    assert manifest["train_sampling"]["target_per_bucket"] == 5
    assert set(manifest["train_sampling"]["after"].values()) == {5}
    assert len(train_rows) == sum(manifest["train_sampling"]["after"].values())
    assert len(manifest["train_sampling"]["after"]) >= 2
    assert manifest["split_counts"]["val"] != len(train_rows)
    assert manifest["split_counts"]["holdout"] != len(train_rows)


def test_phase2af_candidate_order_variants_preserve_evidence_without_oracle_markers() -> None:
    row = _row(7, expected_slot=2, visible_symbol_slot=0, split="train")
    row["expected_patch_candidate_slot"] = 2

    variants = _candidate_order_variants(row)

    assert { _expected_slot_for_manifest(variant) for variant in variants } == {0, 1, 2, 3}
    assert {variant["current_visible_text"] for variant in variants} == {row["current_visible_text"]}
    assert {json.dumps(variant["runtime_visible_evidence"], sort_keys=True) for variant in variants} == {
        json.dumps(row["runtime_visible_evidence"], sort_keys=True)
    }
    assert {variant["expected_repair_action"] for variant in variants} == {row["expected_repair_action"]}
    augmented = [variant for variant in variants if "phase2af_train_augmentation" in variant]
    assert augmented
    assert all(
        variant["phase2af_train_augmentation"]["type"] == "candidate_order_invariance"
        for variant in augmented
    )
    assert all(variant["phase2af_train_augmentation"]["sealed_feedback_used"] is False for variant in augmented)


def test_phase2af_expected_slot_prefers_current_candidate_order_over_stale_slot() -> None:
    row = _row(9, expected_slot=2, visible_symbol_slot=0, split="train")
    row["expected_patch_candidate_slot"] = 2
    variant = [
        item
        for item in _candidate_order_variants(row)
        if item.get("phase2af_train_augmentation", {}).get("augmented_expected_slot") == 0
    ][0]

    assert variant["expected_patch_candidate_slot"] == 2
    assert _expected_slot_for_manifest(variant) == 0


def test_phase2af_stratified_builder_records_candidate_order_augmentation(
    tmp_path: Path,
) -> None:
    rows = [_row(index, expected_slot=1, visible_symbol_slot=0, split="train") for index in range(8)]
    rows.extend(
        _row(100 + index, expected_slot=2, visible_symbol_slot=0, split="train") for index in range(4)
    )
    rows.extend(
        _row(200 + index, expected_slot=1, visible_symbol_slot=1, split="train") for index in range(12)
    )
    source = _write_jsonl(tmp_path / "rows.jsonl", rows)

    manifest = build_phase2af_stratified_hardened_structural_sidecar_split(
        train_jsonls=[source],
        val_jsonls=[source],
        holdout_jsonls=[source],
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
        min_train_rows=12,
        min_val_rows=12,
        min_holdout_rows=12,
        max_train_rows=24,
        max_val_rows=24,
        max_holdout_rows=24,
        augment_train_candidate_order=True,
    )

    assert manifest["train_augmentation"]["candidate_order_invariance_train_only"] is True
    assert manifest["train_augmentation"]["rows_after"] > manifest["train_augmentation"]["rows_before"]
    assert manifest["train_augmentation"]["sealed_feedback_used"] is False
    assert manifest["train_augmentation"]["runtime_visible_evidence_unchanged"] is True


def test_phase2af_stratified_builder_can_balance_train_slots(tmp_path: Path) -> None:
    rows = [_row(index, expected_slot=1, visible_symbol_slot=0, split="train") for index in range(8)]
    rows.extend(
        _row(100 + index, expected_slot=2, visible_symbol_slot=0, split="train") for index in range(4)
    )
    rows.extend(
        _row(200 + index, expected_slot=1, visible_symbol_slot=1, split="train") for index in range(12)
    )
    source = _write_jsonl(tmp_path / "rows.jsonl", rows)

    manifest = build_phase2af_stratified_hardened_structural_sidecar_split(
        train_jsonls=[source],
        val_jsonls=[source],
        holdout_jsonls=[source],
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
        min_train_rows=12,
        min_val_rows=12,
        min_holdout_rows=12,
        max_train_rows=24,
        max_val_rows=24,
        max_holdout_rows=24,
        augment_train_candidate_order=True,
        balance_train_slots=True,
        train_slot_target=6,
    )
    train_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert manifest["train_slot_sampling"]["slot_balanced_train"] is True
    assert manifest["train_slot_sampling"]["after"] == {"0": 6, "1": 6, "2": 6, "3": 6}
    assert len(train_rows) == 24
    assert "train_slot_sampling" in manifest
    assert manifest["train_augmentation"]["candidate_order_invariance_train_only"] is True
