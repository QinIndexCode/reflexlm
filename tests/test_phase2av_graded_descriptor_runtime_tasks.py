import json
from pathlib import Path

from reflexlm.cli.audit_phase2av_graded_descriptor_runtime_tasks import (
    audit_phase2av_graded_descriptor_runtime_tasks,
)
from reflexlm.cli.build_phase2av_graded_descriptor_runtime_tasks import (
    build_phase2av_graded_descriptor_runtime_tasks,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _write_test(path: Path, *, parser_oracle: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if parser_oracle:
        path.write_text(
            "from pathlib import Path\n\n"
            "def test_parser_oracle():\n"
            "    text = Path('src/module.py').read_text(encoding='utf-8')\n"
            "    assert 'exact required text' in text\n",
            encoding="utf-8",
        )
        return
    path.write_text(
        "def test_runtime_behavior():\n"
        "    observed = {'before': False, 'after': True, 'rollback_clean': True}\n"
        "    assert observed['after'] is True\n"
        "    assert observed['rollback_clean'] is True\n",
        encoding="utf-8",
    )


def _source_row(
    index: int,
    *,
    operation: str,
    template: str,
    repo_index: int | None = None,
    parser_oracle: bool = False,
    exception_text: str | None = None,
) -> dict:
    repo_number = index if repo_index is None else repo_index
    return {
        "trace_id": f"holdout:repo-{repo_number}:{index}",
        "benchmark_family": "phase2at_learned_bounded_patch_candidate_generation",
        "claim_boundary": "phase2at_learned_bounded_patch_candidate_generation_pretrain_only",
        "source_kind": "public_repo",
        "repo_url_or_origin": f"https://github.com/example/repo-{repo_number}.git",
        "commit_hash": "a" * 40,
        "current_visible_text": "Public runtime repair evidence only.",
        "runtime_visible_evidence": {
            "changed_files": ["src/module.py"],
            "repair_modes": [template],
            "structural_probe_hashes": [f"probe-{index}"],
            "pytest_before_patch": {"stdout_excerpt": exception_text or "AssertionError"},
        },
        "difficulty": {
            "candidate_count": 2,
            "evidence_density": "medium",
            "repair_depth": "two_edits",
        },
        "repair_candidates": [
            {
                "repair_action": f"repair-{index}-a",
                "intent": "apply_patch_and_rerun_tests",
                "structural_probe_hash": f"probe-{index}",
                "target_symbol": f"symbol-{index}",
            },
            {
                "repair_action": f"repair-{index}-b",
                "intent": "apply_patch_and_rerun_tests",
                "structural_probe_hash": f"distractor-{index}",
                "target_symbol": f"distractor-symbol-{index}",
            },
        ],
        "expected_repair_action": f"repair-{index}-a",
        "artifact_paths": {
            "generated_test": f"generated/test_{index}_{'oracle' if parser_oracle else 'runtime'}.py"
        },
        "learned_patch_candidate_target": {
            "schema_version": "phase2at.learned_bounded_patch_candidate.v1",
            "target_source": "runtime_visible_structural_descriptor_not_recorded_patch",
            "target_path": "src/module.py",
            "operation": operation,
            "anchor": {"kind": "runtime_structural_probe", "probe_hash": f"probe-{index}"},
            "before_fragment_hash": "b" * 16,
            "after_fragment_template_id": template,
            "literal_or_symbol_payload": {"target_symbol_hash": f"symbol-{index}"},
            "safety_constraints": {
                "allowed_paths": ["src/module.py"],
                "forbid_unbounded_diff_text": True,
                "require_anchor_match": True,
                "require_rollback_verification": True,
            },
            "verification_command_slot": 0,
        },
        "freeform_patch_generation": False,
        "recorded_patch_artifact_as_generation_target": False,
        "symbolic_generator_as_generation_target": False,
        "sealed_feedback_used": False,
    }


def _build_rows(tmp_path: Path, rows: list[dict]) -> Path:
    for row in rows:
        test_rel = row["artifact_paths"]["generated_test"]
        _write_test(
            tmp_path / test_rel,
            parser_oracle=test_rel.endswith("_oracle.py"),
        )
    source = _write_jsonl(tmp_path / "source.jsonl", rows)
    report = build_phase2av_graded_descriptor_runtime_tasks(
        input_jsonl=source,
        source_dataset_root=tmp_path,
        output_jsonl=tmp_path / "phase2av" / "holdout.jsonl",
        split="holdout",
        min_rows=len(rows),
    )
    assert report["converted_row_count"] == len(rows)
    return tmp_path / "phase2av" / "holdout.jsonl"


def test_phase2av_accepts_diverse_non_parser_oracle_runtime_tasks(tmp_path: Path) -> None:
    rows = [
        _source_row(0, operation="insert_import", template="import_restoration"),
        _source_row(1, operation="replace_attribute", template="call_attribute_restoration"),
    ]
    tasks = _build_rows(tmp_path, rows)

    report = audit_phase2av_graded_descriptor_runtime_tasks(
        tasks_jsonl=tasks,
        dataset_root=tmp_path / "phase2av",
        min_rows=2,
        min_repo_origins=2,
        min_operation_template_pairs=2,
    )

    assert report["passed"] is True
    assert report["checks"]["operation_template_diversity_met"] is True
    assert report["checks"]["generated_tests_not_parser_oracle_solvable"] is True


def test_phase2av_rejects_single_template_runtime_family(tmp_path: Path) -> None:
    rows = [
        _source_row(0, operation="insert_import", template="import_restoration", repo_index=0),
        _source_row(1, operation="insert_import", template="import_restoration", repo_index=1),
    ]
    tasks = _build_rows(tmp_path, rows)

    report = audit_phase2av_graded_descriptor_runtime_tasks(
        tasks_jsonl=tasks,
        dataset_root=tmp_path / "phase2av",
        min_rows=2,
        min_repo_origins=2,
        min_operation_template_pairs=2,
    )

    assert report["passed"] is False
    assert report["checks"]["operation_template_diversity_met"] is False
    assert "do_not_train_phase2av" in report["blocked_actions"]


def test_phase2av_rejects_parser_oracle_generated_tests(tmp_path: Path) -> None:
    rows = [
        _source_row(0, operation="insert_import", template="import_restoration"),
        _source_row(
            1,
            operation="replace_attribute",
            template="call_attribute_restoration",
            parser_oracle=True,
        ),
    ]
    tasks = _build_rows(tmp_path, rows)

    report = audit_phase2av_graded_descriptor_runtime_tasks(
        tasks_jsonl=tasks,
        dataset_root=tmp_path / "phase2av",
        min_rows=2,
        min_repo_origins=2,
        min_operation_template_pairs=2,
    )

    assert report["passed"] is False
    assert report["checks"]["generated_tests_not_parser_oracle_solvable"] is False
    assert report["metrics"]["parser_oracle_rows"]


def test_phase2av_builder_rejects_exception_operation_inconsistent_rows(
    tmp_path: Path,
) -> None:
    rows = [
        _source_row(
            0,
            operation="replace_attribute",
            template="call_attribute_restoration",
            exception_text="NameError: name 'pluggy' is not defined",
        ),
        _source_row(
            1,
            operation="insert_import",
            template="import_restoration",
            exception_text="AttributeError: 'str' object has no attribute 'join'",
        ),
    ]
    for row in rows:
        _write_test(tmp_path / row["artifact_paths"]["generated_test"])
    source = _write_jsonl(tmp_path / "source.jsonl", rows)

    report = build_phase2av_graded_descriptor_runtime_tasks(
        input_jsonl=source,
        source_dataset_root=tmp_path,
        output_jsonl=tmp_path / "phase2av" / "holdout.jsonl",
        split="holdout",
        min_rows=2,
    )

    assert report["passed"] is False
    assert report["converted_row_count"] == 0
    assert report["reject_counts"] == {
        "attribute_error_labeled_as_import_restoration": 1,
        "name_error_labeled_as_attribute_restoration": 1,
    }


def test_phase2av_audit_rejects_exception_operation_inconsistent_tasks(
    tmp_path: Path,
) -> None:
    rows = [
        _source_row(0, operation="insert_import", template="import_restoration"),
        _source_row(1, operation="replace_attribute", template="call_attribute_restoration"),
    ]
    tasks = _build_rows(tmp_path, rows)
    task_rows = [
        json.loads(line) for line in tasks.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    task_rows[1]["runtime_visible_evidence"]["pytest_before_patch"] = {
        "stdout_excerpt": "NameError: name 'pluggy' is not defined"
    }
    _write_jsonl(tasks, task_rows)

    report = audit_phase2av_graded_descriptor_runtime_tasks(
        tasks_jsonl=tasks,
        dataset_root=tmp_path / "phase2av",
        min_rows=2,
        min_repo_origins=2,
        min_operation_template_pairs=2,
    )

    assert report["passed"] is False
    assert report["checks"]["operation_exception_evidence_consistent"] is False
    assert report["metrics"]["exception_inconsistent_rows"] == [
        {
            "task_id": "phase2av:holdout:00001",
            "reason": "name_error_labeled_as_attribute_restoration",
        }
    ]


def test_phase2av_rejects_sealed_or_recorded_target_rows(tmp_path: Path) -> None:
    sealed = _source_row(0, operation="insert_import", template="import_restoration")
    sealed["sealed_feedback_used"] = True
    recorded = _source_row(1, operation="replace_attribute", template="call_attribute_restoration")
    recorded["recorded_patch_artifact_as_generation_target"] = True
    source = _write_jsonl(tmp_path / "source.jsonl", [sealed, recorded])

    report = build_phase2av_graded_descriptor_runtime_tasks(
        input_jsonl=source,
        source_dataset_root=tmp_path,
        output_jsonl=tmp_path / "phase2av" / "holdout.jsonl",
        split="holdout",
        min_rows=2,
    )

    assert report["passed"] is False
    assert report["converted_row_count"] == 0
    assert report["reject_counts"] == {
        "missing_generated_tests": 2,
        "recorded_patch_target_not_disabled": 1,
        "sealed_feedback_not_absent": 1,
    }
