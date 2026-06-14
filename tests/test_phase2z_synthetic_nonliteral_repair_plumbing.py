import json
from pathlib import Path

from reflexlm.cli.audit_phase2z_synthetic_nonliteral_repair_plumbing import (
    audit_phase2z_synthetic_nonliteral_repair_plumbing,
)
from reflexlm.cli.run_phase2z_synthetic_nonliteral_repair_plumbing import (
    run_phase2z_synthetic_nonliteral_repair_plumbing,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _synthetic_dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "phase2s"
    repo = root / "source_repos" / "holdout" / "repo0"
    (repo / "tests").mkdir(parents=True)
    (repo / "repair_case.py").write_text(
        "from repair_helper import transform\n\n"
        "SUFFIX = 'broken'\n\n"
        "def evaluate(value):\n"
        "    return transform(value) + '-' + SUFFIX\n",
        encoding="utf-8",
    )
    (repo / "repair_helper.py").write_text(
        "def transform(value):\n"
        "    return str(value).strip().upper()\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_repair_case.py").write_text(
        "from repair_case import evaluate\n\n\n"
        "def test_repair_case_behavior():\n"
        "    assert evaluate('  case001  ') == 'case001-fixed'\n",
        encoding="utf-8",
    )
    artifact = root / "artifacts" / "holdout" / "repo0"
    artifact.mkdir(parents=True)
    patch = (
        "--- a/repair_case.py\n"
        "+++ b/repair_case.py\n"
        "@@ -1,6 +1,6 @@\n"
        " from repair_helper import transform\n"
        " \n"
        "-SUFFIX = 'broken'\n"
        "+SUFFIX = 'fixed'\n"
        " \n"
        " def evaluate(value):\n"
        "     return transform(value) + '-' + SUFFIX\n"
        "--- a/repair_helper.py\n"
        "+++ b/repair_helper.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def transform(value):\n"
        "-    return str(value).strip().upper()\n"
        "+    return str(value).strip().lower()\n"
    )
    (artifact / "patch.diff").write_text(patch, encoding="utf-8")
    rows = [
        {
            "trace_id": "holdout:repo0:0",
            "split": "holdout",
            "source_kind": "synthetic_safe_repo",
            "repo_id": "repo0",
            "repo_url_or_origin": "synthetic://phase2s/repo0",
            "commit_hash": "a" * 40,
            "runtime_visible_evidence": {
                "changed_files": ["repair_case.py", "repair_helper.py"],
            },
            "repair_candidates": [
                {
                    "verification_command": "python -m pytest -q tests/test_repair_case.py --maxfail=1"
                }
            ],
            "expected_repair_result": {"test_target": "tests/test_repair_case.py"},
            "artifact_paths": {"patch_diff": "artifacts/holdout/repo0/patch.diff"},
        }
    ]
    return root, _write_jsonl(tmp_path / "rows.jsonl", rows)


def test_phase2z_runner_records_synthetic_nonclaim_nonliteral_success(tmp_path: Path) -> None:
    dataset_root, rows_jsonl = _synthetic_dataset(tmp_path)
    package = _write_json(
        tmp_path / "package" / "native_nervous_package.json",
        {
            "policy_label": "test-policy",
            "open_repair_capabilities": {
                "patch_proposal": True,
                "test_selection": True,
                "rollback_safety": True,
                "bounded_edit_scope": True,
                "progress_monitor": True,
                "verification_state": True,
                "stop_condition": True,
            },
        },
    )

    report = run_phase2z_synthetic_nonliteral_repair_plumbing(
        source_rows_jsonl=rows_jsonl,
        dataset_root=dataset_root,
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
    )

    assert report["successes"] == 1
    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert result["claim_bearing_execution_evidence"] is False
    assert result["oracle_trace_used"] is True
    assert result["patch_stats"]["multi_file"] is True
    assert result["patch_stats"]["nonliteral_structure_present"] is True


def test_phase2z_audit_keeps_plumbing_out_of_claim_boundary(tmp_path: Path) -> None:
    rows = [
        {
            "success": True,
            "claim_bearing_execution_evidence": False,
            "recorded_patch_artifact_used": True,
            "oracle_trace_used": True,
            "sealed_feedback_used": False,
            "patch_stats": {"nonliteral_structure_present": True, "multi_file": True},
        }
    ]
    results = _write_jsonl(tmp_path / "results.jsonl", rows)

    report = audit_phase2z_synthetic_nonliteral_repair_plumbing(
        execution_results_jsonl=results,
        min_rows=1,
        min_success_rate=1.0,
    )

    assert report["passed"] is True
    assert report["claim_bearing_execution_evidence"] is False
    assert "do_not_use_phase2z_plumbing_as_open_ended_debugging_claim" in report["blocked_actions"]


def test_phase2z_audit_rejects_claim_bearing_recorded_patch_rows(tmp_path: Path) -> None:
    rows = [
        {
            "success": True,
            "claim_bearing_execution_evidence": True,
            "recorded_patch_artifact_used": True,
            "oracle_trace_used": True,
            "sealed_feedback_used": False,
            "patch_stats": {"nonliteral_structure_present": True, "multi_file": True},
        }
    ]
    results = _write_jsonl(tmp_path / "results.jsonl", rows)

    report = audit_phase2z_synthetic_nonliteral_repair_plumbing(
        execution_results_jsonl=results,
        min_rows=1,
        min_success_rate=1.0,
    )

    assert report["passed"] is False
    assert report["checks"]["all_rows_non_claim_plumbing"] is False
