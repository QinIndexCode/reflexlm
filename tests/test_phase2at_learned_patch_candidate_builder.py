import json
from pathlib import Path

from reflexlm.cli.audit_phase2at_learned_patch_candidate_data import (
    SCHEMA_VERSION,
    audit_phase2at_learned_patch_candidate_data,
)
from reflexlm.cli.build_phase2at_learned_patch_candidate_data import (
    build_phase2at_learned_patch_candidate_data,
    phase2z_row_to_phase2at,
)
from reflexlm.cli.build_phase2s_head_dataset import phase2s_repair_trace_to_head_row
from reflexlm.llm.native_cortex import PATCH_OPERATION_ORDER, PATCH_TEMPLATE_ORDER


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _phase2z_row(index: int, split: str) -> dict:
    expected = f"structural_repair_{index % 2}"
    return {
        "trace_id": f"{split}:repo{index % 2}:{index}",
        "split": split,
        "source_kind": "public_repo",
        "repo_id": f"{split}_repo_{index % 3}",
        "repo_url_or_origin": f"https://example.invalid/{split}_repo_{index % 3}.git",
        "current_visible_text": "public runtime evidence without marker leakage",
        "runtime_visible_evidence": {
            "changed_files": [f"pkg/module_{index % 3}.py"],
            "watched_files": [f"tests/test_{index % 3}.py"],
            "repair_modes": ["call_attribute_restoration" if index % 2 else "import_restoration"],
            "structural_probe_hashes": [f"probe-{index % 2}"],
            "pytest_before_patch": {"stdout_excerpt": f"failure {index}"},
        },
        "repair_candidates": [
            {
                "repair_action": "structural_repair_0",
                "structural_probe_hash": "probe-0",
                "target_symbol": "sym-0",
            },
            {
                "repair_action": "structural_repair_1",
                "structural_probe_hash": "probe-1",
                "target_symbol": "sym-1",
            },
        ],
        "expected_repair_action": expected,
        "baselines": {
            "source_overlap": expected if index % 4 == 0 else "structural_repair_0",
            "prompt_only": "structural_repair_0",
        },
        "artifact_paths": {
            "patch_diff": f"artifacts/{split}/repo/row_{index:05d}/patch.diff",
            "generated_test": f"artifacts/{split}/repo/row_{index:05d}/generated_test.py",
        },
        "normalization": {"sealed_feedback_absent": True},
    }


def test_phase2at_builder_converts_phase2z_row_to_bounded_descriptor() -> None:
    converted = phase2z_row_to_phase2at(_phase2z_row(1, "val"))

    target = converted["learned_patch_candidate_target"]
    assert target["schema_version"] == SCHEMA_VERSION
    assert target["operation"] == "replace_attribute"
    assert target["after_fragment_template_id"] == "call_attribute_restoration"
    assert target["target_source"] == "runtime_visible_structural_descriptor_not_recorded_patch"
    assert "patch_diff" not in converted["artifact_paths"]
    assert converted["recorded_patch_artifact_as_generation_target"] is False
    assert converted["symbolic_generator_as_generation_target"] is False


def test_phase2at_descriptor_labels_are_carried_into_head_row() -> None:
    converted = phase2z_row_to_phase2at(_phase2z_row(1, "val"))

    head_row = phase2s_repair_trace_to_head_row(converted)

    assert head_row["patch_operation_label"] == PATCH_OPERATION_ORDER.index("replace_attribute")
    assert head_row["patch_target_file_slot"] == 0
    assert head_row["patch_template_slot"] == PATCH_TEMPLATE_ORDER.index(
        "call_attribute_restoration"
    )


def test_phase2at_builder_uses_primary_runtime_repair_mode_for_template() -> None:
    row = _phase2z_row(1, "val")
    row["runtime_visible_evidence"]["repair_modes"] = [
        "import_restoration",
        "call_attribute_restoration",
    ]

    converted = phase2z_row_to_phase2at(row)
    head_row = phase2s_repair_trace_to_head_row(converted)

    assert converted["learned_patch_candidate_target"]["operation"] == "insert_import"
    assert converted["learned_patch_candidate_target"]["after_fragment_template_id"] == (
        "import_restoration"
    )
    assert head_row["patch_operation_label"] == PATCH_OPERATION_ORDER.index("insert_import")
    assert head_row["patch_template_slot"] == PATCH_TEMPLATE_ORDER.index("import_restoration")


def test_phase2at_builder_maps_behavioral_runtime_modes_to_descriptor_labels() -> None:
    import_row = _phase2z_row(0, "val")
    import_row["runtime_visible_evidence"]["repair_modes"] = [
        "behavioral_import_restoration"
    ]
    method_row = _phase2z_row(1, "val")
    method_row["runtime_visible_evidence"]["repair_modes"] = [
        "behavioral_string_method_restoration"
    ]

    converted_import = phase2z_row_to_phase2at(import_row)
    converted_method = phase2z_row_to_phase2at(method_row)

    assert converted_import["learned_patch_candidate_target"]["operation"] == "insert_import"
    assert converted_import["learned_patch_candidate_target"]["after_fragment_template_id"] == (
        "import_restoration"
    )
    assert converted_method["learned_patch_candidate_target"]["operation"] == (
        "replace_attribute"
    )
    assert converted_method["learned_patch_candidate_target"]["after_fragment_template_id"] == (
        "call_attribute_restoration"
    )


def test_phase2at_builder_maps_literal_runtime_mode_to_descriptor_labels() -> None:
    row = _phase2z_row(2, "val")
    row["runtime_visible_evidence"]["repair_modes"] = [
        "module_constant_literal_restoration"
    ]

    converted = phase2z_row_to_phase2at(row)
    head_row = phase2s_repair_trace_to_head_row(converted)

    assert converted["learned_patch_candidate_target"]["operation"] == "replace_literal"
    assert converted["learned_patch_candidate_target"]["after_fragment_template_id"] == (
        "literal_restoration"
    )
    assert head_row["patch_operation_label"] == PATCH_OPERATION_ORDER.index(
        "replace_literal"
    )
    assert head_row["patch_template_slot"] == PATCH_TEMPLATE_ORDER.index(
        "literal_restoration"
    )


def test_phase2at_builder_outputs_split_that_passes_data_gate(tmp_path: Path) -> None:
    source_paths = {}
    for split in ("train", "val", "holdout"):
        source_paths[split] = _write_jsonl(
            tmp_path / f"{split}.raw.jsonl",
            [_phase2z_row(index, split) for index in range(24)],
        )

    manifest = build_phase2at_learned_patch_candidate_data(
        train_jsonl=source_paths["train"],
        val_jsonl=source_paths["val"],
        holdout_jsonl=source_paths["holdout"],
        output_dir=tmp_path / "phase2at",
        manifest_json=tmp_path / "manifest.json",
    )
    audit = audit_phase2at_learned_patch_candidate_data(
        train_jsonl=tmp_path / "phase2at" / "train.jsonl",
        val_jsonl=tmp_path / "phase2at" / "val.jsonl",
        holdout_jsonl=tmp_path / "phase2at" / "holdout.jsonl",
    )

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["recorded_patch_artifact_as_generation_target"] is False
    assert audit["passed"] is True
    assert audit["checks"]["no_patch_diff_artifact_available_as_training_target"] is True


def test_phase2at_data_gate_rejects_patch_diff_artifact_even_with_valid_target(
    tmp_path: Path,
) -> None:
    rows = [phase2z_row_to_phase2at(_phase2z_row(index, "val")) for index in range(24)]
    for row in rows:
        row["artifact_paths"]["patch_diff"] = "artifacts/forbidden.patch"
    train = _write_jsonl(tmp_path / "train.jsonl", rows)
    val = _write_jsonl(tmp_path / "val.jsonl", rows)
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", rows)

    report = audit_phase2at_learned_patch_candidate_data(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
    )

    assert report["passed"] is False
    assert report["checks"]["no_patch_diff_artifact_available_as_training_target"] is False
