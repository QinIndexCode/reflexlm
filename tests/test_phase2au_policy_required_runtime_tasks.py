import json
from pathlib import Path

from reflexlm.cli.audit_phase2au_policy_required_runtime_tasks import (
    audit_phase2au_policy_required_runtime_tasks,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _write_test(root: Path, rel: str, *, parser_oracle: bool = False) -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if parser_oracle:
        path.write_text(
            "from pathlib import Path\n\n"
            "def test_parser_oracle():\n"
            "    text = Path('src/module.py').read_text(encoding='utf-8')\n"
            "    assert 'exact_required_text' in text\n",
            encoding="utf-8",
        )
    else:
        path.write_text(
            "def test_semantic_relation_runtime_visible():\n"
            "    observed = {'before': 2, 'after': 3, 'rollback_clean': True}\n"
            "    assert observed['after'] > observed['before']\n"
            "    assert observed['rollback_clean'] is True\n",
            encoding="utf-8",
        )
    return rel


def _row(index: int, root: Path, *, parser_oracle: bool = False) -> dict:
    axes = [
        "ambiguous_nonliteral_semantic",
        "multi_file_interaction",
        "negative_constraint",
        "stateful_verification",
    ]
    test_rel = _write_test(
        root,
        f"tests/generated/test_phase2au_{index}.py",
        parser_oracle=parser_oracle,
    )
    return {
        "task_id": f"phase2au:holdout:{index:05d}",
        "benchmark_family": "phase2au_policy_required_runtime_delta",
        "split": "holdout",
        "repo_origin": f"https://github.com/example/repo-{index % 4}.git",
        "repo_commit": "a" * 40,
        "problem_statement": "Select a bounded learned patch descriptor from runtime-visible evidence.",
        "evaluation_commands": [f"python -m pytest -q {test_rel} --maxfail=1"],
        "artifact_paths": {"generated_test": test_rel},
        "allowed_write_scope": ["src/module.py", "src/helpers.py"],
        "difficulty_axes": axes,
        "runtime_visible_contract": {
            "policy_required_runtime_delta": True,
            "no_policy_symbolic_control_expected_to_fail": True,
            "no_direct_text_membership_or_ast_attr_oracle": True,
            "no_candidate_slot_marker": True,
            "no_gold_hint": True,
        },
        "expected_policy": {
            "patch_proposal": 1,
            "patch_operation": "replace_symbol",
            "patch_template": "symbol_reference_restoration",
            "bounded_edit_scope": 1,
            "rollback_safety": 1,
        },
        "candidate_policy_commands": [
            "phase2au_apply_candidate --repair-action repair_a --verify pytest",
            "phase2au_apply_candidate --repair-action repair_b --verify pytest",
        ],
        "expected_repair_action": "repair_a",
        "sealed_feedback_used": False,
        "task_spec_sha256": "b" * 64,
    }


def test_phase2au_policy_required_gate_accepts_non_parser_oracle_tasks(
    tmp_path: Path,
) -> None:
    rows = [_row(index, tmp_path) for index in range(64)]
    tasks = _write_jsonl(tmp_path / "tasks.jsonl", rows)

    report = audit_phase2au_policy_required_runtime_tasks(
        tasks_jsonl=tasks,
        dataset_root=tmp_path,
    )

    assert report["passed"] is True
    assert report["checks"]["generated_tests_not_parser_oracle_solvable"] is True
    assert report["metrics"]["repo_origin_count"] == 4


def test_phase2au_policy_required_gate_rejects_parser_oracle_tests(
    tmp_path: Path,
) -> None:
    rows = [_row(index, tmp_path, parser_oracle=(index == 0)) for index in range(64)]
    tasks = _write_jsonl(tmp_path / "tasks.jsonl", rows)

    report = audit_phase2au_policy_required_runtime_tasks(
        tasks_jsonl=tasks,
        dataset_root=tmp_path,
    )

    assert report["passed"] is False
    assert report["checks"]["generated_tests_not_parser_oracle_solvable"] is False
    assert report["metrics"]["parser_oracle_rows"] == ["phase2au:holdout:00000"]


def test_phase2au_policy_required_gate_rejects_missing_policy_contract(
    tmp_path: Path,
) -> None:
    rows = [_row(index, tmp_path) for index in range(64)]
    rows[0]["runtime_visible_contract"]["policy_required_runtime_delta"] = False
    rows[1]["expected_policy"].pop("patch_template")
    tasks = _write_jsonl(tmp_path / "tasks.jsonl", rows)

    report = audit_phase2au_policy_required_runtime_tasks(
        tasks_jsonl=tasks,
        dataset_root=tmp_path,
    )

    assert report["passed"] is False
    assert report["checks"]["all_rows_policy_required_contract"] is False
    assert report["checks"]["expected_policy_requires_learned_patch_heads"] is False
