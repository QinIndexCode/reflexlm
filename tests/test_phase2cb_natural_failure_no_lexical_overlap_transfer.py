import json

from reflexlm.cli.run_phase2cb_natural_failure_no_lexical_overlap_transfer import (
    NATURAL_INTENT_COMMANDS,
    _eligible_rows,
    _lexical_overlap,
    _manifest,
)


def _row(task_id: str, operation: str, failure: str) -> dict:
    return {
        "task_id": task_id,
        "source_kind": "public_repo",
        "repo_origin": "https://example.test/repo.git",
        "repo_commit": "abc123",
        "learned_patch_descriptor_target": {"operation": operation},
        "runtime_visible_contract": {"no_gold_hint": True},
        "runtime_visible_evidence": {
            "changed_files": ["src/example.py"],
            "pytest_before_patch": {"stdout_excerpt": f"E   {failure}\n"},
        },
    }


def test_phase2cb_filters_natural_zero_overlap_rows(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [
        _row(
            "ok",
            "replace_attribute",
            "AttributeError: 'str' object has no attribute 'phase2z_missing_strip'",
        ),
        _row("overlap", "insert_import", "NameError: required library missing"),
        {**_row("hidden", "replace_literal", "AssertionError: assert 1 == 2"), "source_kind": "synthetic"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    filtered = _eligible_rows(path)

    assert [row["task_id"] for row in filtered] == ["ok"]
    assert _lexical_overlap(filtered[0]) == []


def test_phase2cb_manifest_has_erased_and_wrong_receptor_controls(tmp_path) -> None:
    rows = [
        _row(
            "attribute",
            "replace_attribute",
            "AttributeError: 'str' object has no attribute 'phase2z_missing_strip'",
        ),
        _row("import", "insert_import", "NameError: name 'os' is not defined"),
        _row("literal", "replace_literal", "AssertionError: assert 89 == 88"),
    ]

    visible = _manifest(rows=rows, workspace_root=tmp_path, suite_id="visible")
    erased = _manifest(rows=rows, workspace_root=tmp_path, suite_id="erased")
    wrong = _manifest(rows=rows, workspace_root=tmp_path, suite_id="wrong_receptor")

    for manifest in [visible, erased, wrong]:
        assert len(manifest["episodes"]) == len(rows)
        for episode in manifest["episodes"]:
            contract = episode["natural_failure_contract"]
            assert contract["correct_candidate_is_first"] is False
            assert contract["failure_correct_command_overlap"] == []
            assert contract["free_form_action_generation"] is False
            assert "expected_sequence" not in episode and "steps" not in episode
    assert (
        visible["episodes"][0]["initial_state"]["terminal"]["stderr_delta"]
        != erased["episodes"][0]["initial_state"]["terminal"]["stderr_delta"]
    )
    assert visible["episodes"][0]["initial_state"]["filesystem"]["changed_paths"] == []
    assert visible["episodes"][0]["initial_state"]["filesystem"]["dirty_files"] == []
    assert visible["episodes"][0]["initial_state"]["filesystem"]["external_change_detected"] is False
    assert visible["episodes"][0]["initial_state"]["runtime_evidence"][
        "terminal_observations"
    ] == ["src/example.py"]
    assert wrong["episodes"][0]["natural_failure_contract"]["wrong_receptor_source_task_id"] == "import"
    assert set(NATURAL_INTENT_COMMANDS) == {
        "replace_attribute",
        "insert_import",
        "replace_literal",
    }
