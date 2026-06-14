import json
from pathlib import Path

from reflexlm.cli.build_phase2au_policy_required_runtime_tasks import (
    build_phase2au_policy_required_runtime_tasks,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _base_row(index: int, *, test_rel: str = "generated/test_runtime.py") -> dict:
    return {
        "task_id": f"phase2y:holdout:{index:05d}",
        "benchmark_family": "phase2y_open_repair_generalization_pressure",
        "split": "holdout",
        "repo_origin": f"https://github.com/example/repo-{index % 4}.git",
        "repo_commit": "a" * 40,
        "problem_statement": "Resolve a non-parser-oracle runtime repair task.",
        "evaluation_commands": [f"python -m pytest -q {test_rel} --maxfail=1"],
        "artifact_paths": {"generated_test": test_rel},
        "allowed_write_scope": ["src/module.py", "src/helpers.py"],
        "difficulty_axes": [
            "ambiguous_nonliteral_semantic",
            "multi_file_interaction",
            "negative_constraint",
            "stateful_verification",
        ],
        "runtime_visible_contract": {
            "no_candidate_slot_marker": True,
            "no_gold_hint": True,
            "no_sealed_feedback": True,
        },
        "expected_policy": {
            "patch_proposal": 1,
            "patch_operation": "replace_symbol",
            "patch_template": "symbol_reference_restoration",
            "bounded_edit_scope": 1,
            "rollback_safety": 1,
        },
        "repair_candidates": [
            {"repair_action": "repair_a", "intent": "apply_patch_and_rerun_tests"},
            {"repair_action": "repair_b", "intent": "apply_patch_and_rerun_tests"},
        ],
        "expected_repair_action": "repair_a",
        "sealed_feedback_used": False,
        "task_spec_sha256": "b" * 64,
    }


def test_phase2au_builder_converts_non_parser_oracle_candidates(tmp_path: Path) -> None:
    test_rel = "generated/test_runtime.py"
    test_path = tmp_path / test_rel
    test_path.parent.mkdir(parents=True)
    test_path.write_text(
        "def test_behavioral_runtime_relation():\n"
        "    observed = {'before': 1, 'after': 2, 'rollback_clean': True}\n"
        "    assert observed['after'] > observed['before']\n"
        "    assert observed['rollback_clean'] is True\n",
        encoding="utf-8",
    )
    source = _write_jsonl(tmp_path / "source.jsonl", [_base_row(0, test_rel=test_rel)])

    report = build_phase2au_policy_required_runtime_tasks(
        input_tasks_jsonl=source,
        dataset_root=tmp_path,
        output_jsonl=tmp_path / "phase2au.jsonl",
        split="holdout",
        min_rows=1,
    )
    converted = [
        json.loads(line)
        for line in (tmp_path / "phase2au.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert report["passed"] is True
    assert report["converted_row_count"] == 1
    assert converted[0]["benchmark_family"] == "phase2au_policy_required_runtime_delta"
    assert converted[0]["runtime_visible_contract"]["policy_required_runtime_delta"] is True
    assert converted[0]["artifact_paths"]["generated_tests"] == [test_rel]
    assert len(converted[0]["candidate_policy_commands"]) == 2
    assert converted[0]["expected_repair_action"] == "repair_a"


def test_phase2au_builder_rejects_missing_real_test_file(tmp_path: Path) -> None:
    source = _write_jsonl(tmp_path / "source.jsonl", [_base_row(0)])

    report = build_phase2au_policy_required_runtime_tasks(
        input_tasks_jsonl=source,
        dataset_root=tmp_path,
        output_jsonl=tmp_path / "phase2au.jsonl",
        split="holdout",
        min_rows=1,
    )

    assert report["passed"] is False
    assert report["converted_row_count"] == 0
    assert report["reject_counts"] == {"missing_real_generated_test_files": 1}


def test_phase2au_builder_rejects_parser_oracle_and_missing_policy_heads(
    tmp_path: Path,
) -> None:
    test_rel = "generated/test_parser_oracle.py"
    test_path = tmp_path / test_rel
    test_path.parent.mkdir(parents=True)
    test_path.write_text(
        "from pathlib import Path\n\n"
        "def test_parser_oracle():\n"
        "    text = Path('src/module.py').read_text(encoding='utf-8')\n"
        "    assert 'exact_required_text' in text\n",
        encoding="utf-8",
    )
    parser_oracle = _base_row(0, test_rel=test_rel)
    missing_policy = _base_row(1, test_rel=test_rel)
    missing_policy["expected_policy"].pop("patch_template")
    source = _write_jsonl(tmp_path / "source.jsonl", [parser_oracle, missing_policy])

    report = build_phase2au_policy_required_runtime_tasks(
        input_tasks_jsonl=source,
        dataset_root=tmp_path,
        output_jsonl=tmp_path / "phase2au.jsonl",
        split="holdout",
        min_rows=1,
    )

    assert report["passed"] is False
    assert report["converted_row_count"] == 0
    assert report["reject_counts"] == {
        "missing_explicit_learned_patch_policy_heads": 1,
        "parser_oracle_generated_tests": 2,
    }


def test_phase2au_builder_normalizes_raw_behavioral_public_trace(tmp_path: Path) -> None:
    test_rel = "artifacts/holdout/repo/row_00000/generated_test.py"
    test_path = tmp_path / test_rel
    test_path.parent.mkdir(parents=True)
    test_path.write_text(
        "import importlib.util\n"
        "from pathlib import Path\n\n"
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n\n"
        "def _load_module(name, path):\n"
        "    spec = importlib.util.spec_from_file_location(name, path)\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    assert spec is not None and spec.loader is not None\n"
        "    spec.loader.exec_module(module)\n"
        "    return module\n\n"
        "def test_behavioral_repair():\n"
        "    module = _load_module('module_a', REPO_ROOT / 'src/a.py')\n"
        "    assert module is not None\n",
        encoding="utf-8",
    )
    raw = {
        "trace_id": "holdout:repo:phase2z:0",
        "source_kind": "public_repo",
        "synthetic_fault_injected_in_sandbox_only": True,
        "repo_url_or_origin": "https://example.invalid/repo.git",
        "commit_hash": "a" * 40,
        "normalization": {
            "sealed_feedback_absent": True,
            "preserved_runtime_visible_evidence": True,
        },
        "runtime_visible_evidence": {
            "changed_files": ["src/a.py", "src/b.py"],
            "repair_modes": ["behavioral_import_restoration"],
            "structural_probe_hashes": ["probe-a"],
        },
        "repair_runtime": {
            "post_patch_tests_recorded": True,
            "rollback_recorded": True,
            "rollback_failure_recorded": True,
        },
        "difficulty": {"candidate_count": 2},
        "repair_candidates": [
            {"repair_action": "repair_a", "structural_probe_hash": "probe-a"},
            {"repair_action": "repair_b", "structural_probe_hash": "probe-b"},
        ],
        "expected_repair_action": "repair_a",
        "artifact_paths": {"generated_test": test_rel},
        "trace_hash": "c" * 64,
    }
    source = _write_jsonl(tmp_path / "source.jsonl", [raw])

    report = build_phase2au_policy_required_runtime_tasks(
        input_tasks_jsonl=source,
        dataset_root=tmp_path,
        output_jsonl=tmp_path / "phase2au.jsonl",
        split="holdout",
        min_rows=1,
    )
    converted = [
        json.loads(line)
        for line in (tmp_path / "phase2au.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert report["passed"] is True
    assert converted[0]["expected_policy"]["patch_operation"] == "apply_patch_and_rerun_tests"
    assert converted[0]["expected_policy"]["patch_template"] == "behavioral_import_restoration"
    assert len(converted[0]["candidate_policy_commands"]) == 2
    assert converted[0]["runtime_visible_identity"]["command_identity_tokens"] == ["probe-a"]
    assert set(converted[0]["difficulty_axes"]) == {
        "ambiguous_nonliteral_semantic",
        "multi_file_interaction",
        "negative_constraint",
        "stateful_verification",
    }


def test_phase2au_builder_allows_raw_behavioral_trace_without_multifile_axis(
    tmp_path: Path,
) -> None:
    test_rel = "generated/test_behavior.py"
    test_path = tmp_path / test_rel
    test_path.parent.mkdir(parents=True)
    test_path.write_text("def test_behavior():\n    assert 2 > 1\n", encoding="utf-8")
    raw = {
        "trace_id": "holdout:repo:phase2z:single",
        "source_kind": "public_repo",
        "synthetic_fault_injected_in_sandbox_only": True,
        "repo_url_or_origin": "https://example.invalid/repo.git",
        "commit_hash": "a" * 40,
        "normalization": {
            "sealed_feedback_absent": True,
            "preserved_runtime_visible_evidence": True,
        },
        "runtime_visible_evidence": {
            "changed_files": ["src/a.py"],
            "repair_modes": ["behavioral_import_restoration"],
            "structural_probe_hashes": ["probe-a"],
        },
        "repair_runtime": {
            "post_patch_tests_recorded": True,
            "rollback_recorded": True,
            "rollback_failure_recorded": True,
        },
        "difficulty": {"candidate_count": 2},
        "repair_candidates": [
            {"repair_action": "repair_a", "structural_probe_hash": "probe-a"},
            {"repair_action": "repair_b", "structural_probe_hash": "probe-b"},
        ],
        "expected_repair_action": "repair_a",
        "artifact_paths": {"generated_test": test_rel},
        "trace_hash": "c" * 64,
    }
    source = _write_jsonl(tmp_path / "source.jsonl", [raw])

    report = build_phase2au_policy_required_runtime_tasks(
        input_tasks_jsonl=source,
        dataset_root=tmp_path,
        output_jsonl=tmp_path / "phase2au.jsonl",
        split="holdout",
        min_rows=1,
    )

    assert report["passed"] is True
    assert report["converted_row_count"] == 1
