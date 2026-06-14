import json
import subprocess
from pathlib import Path

from reflexlm.cli.collect_phase2t_dynamic_repair_traces import (
    collect_phase2t_dynamic_repair_traces,
    phase2s_row_to_phase2t_repair_loop_row,
)


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_public_literal_repo(tmp_path: Path, repo_id: str) -> Path:
    repo = tmp_path / repo_id
    repo.mkdir(parents=True)
    (repo / "LICENSE").write_text("MIT fixture\n", encoding="utf-8")
    (repo / "module.py").write_text(
        "\n".join(
            [
                "def alpha():",
                "    return 'alpha'",
                "",
                "def bravo():",
                "    return 'bravo'",
                "",
                "def charlie():",
                "    return 3",
                "",
                "def delta():",
                "    return True",
                "",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "phase2t@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Phase2T Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    return repo


def _spec(repo: Path, split: str, repo_id: str, task_family: str) -> dict:
    commit_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    return {
        "repo_id": repo_id,
        "split": split,
        "local_path": str(repo),
        "repo_url": f"https://github.com/example/{repo_id}.git",
        "source_kind": "public_repo",
        "commit_hash": commit_hash,
        "license": "MIT",
        "task_families": [task_family],
        "factor_levels": {
            "candidate_count": ["2", "3", "4"],
            "evidence_density": ["low", "medium", "high"],
            "repair_depth": ["one_edit", "two_edits", "stale_state_refresh"],
            "failure_observability": [
                "direct_traceback",
                "indirect_changed_file_relation",
                "ambiguous_same_intent_command",
            ],
            "ambiguity_class": [
                "same_intent_command",
                "same_file_read",
                "stage_transition",
                "patch_location_ambiguity",
            ],
            "safety_pressure": ["none", "unsafe_command_lure", "rollback_required"],
        },
    }


def test_phase2t_row_transform_adds_repair_loop_schema() -> None:
    phase2s_row = {
        "trace_id": "train:repo:abcdef:phase2s:0",
        "current_visible_text": "visible",
        "runtime_visible_evidence": {},
        "repair_candidates": [{"repair_action": "a"}, {"repair_action": "b"}],
        "repair_runtime": {
            "command_allowlist_observed": True,
            "bounded_edit_scope_observed": True,
            "post_patch_tests_recorded": True,
            "rollback_recorded": True,
            "sandbox_cleanup_recorded": True,
            "source_repo_read_only_observed": True,
        },
        "artifact_paths": {
            "patch_diff": "patch.diff",
            "command_log": "command_log.json",
            "test_output": "test_output.json",
            "rollback_log": "rollback_log.json",
            "sandbox_integrity_report": "sandbox_integrity.json",
        },
    }

    row = phase2s_row_to_phase2t_repair_loop_row(
        phase2s_row,
        spec={
            "task_families": ["false_completion_trap"],
            "factor_levels": {"safety_pressure": ["rollback_required"]},
        },
        row_index=0,
    )

    assert row["phase"] == "Phase2T"
    assert ":phase2t:" in row["trace_id"]
    assert row["claim_bearing_training_ready"] is False
    assert row["difficulty"]["task_family"] == "false_completion_trap"
    assert row["difficulty"]["safety_pressure"] == "rollback_required"
    assert row["repair_loop_episode"]["loop_schema"] == "phase2t_repair_loop_v1"
    assert row["architecture_targets"]["patch_proposal_head"]["required"] is True
    assert row["safety_controls"]["low_level_qwen_calls"] == 0
    assert row["modern_baseline_contract"]["measured_not_declared"] is True


def test_phase2t_dynamic_repair_collector_uses_sandbox_and_keeps_training_blocked(
    tmp_path: Path,
) -> None:
    repo = _make_public_literal_repo(tmp_path, "public_repo")
    specs = [
        _spec(repo, "train", "phase2t_train_repo", "dependency_or_import_mismatch"),
        _spec(repo, "val", "phase2t_val_repo", "localized_unit_assertion"),
        _spec(repo, "holdout", "phase2t_holdout_repo", "false_completion_trap"),
    ]
    specs_path = _write(tmp_path / "specs.json", specs)

    manifest = collect_phase2t_dynamic_repair_traces(
        repo_specs_json=specs_path,
        output_root=tmp_path / "phase2t_traces",
        clone_root=tmp_path / "clones",
        rows_per_repo=1,
        timeout_seconds=10,
        no_clone=True,
    )
    train_row = json.loads(
        (tmp_path / "phase2t_traces" / "train.raw.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    source_status = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True)

    assert manifest["collector_family"] == "phase2t_dynamic_public_repo_repair_loop_trace_collector"
    assert manifest["claim_bearing_training_ready"] is False
    assert manifest["sealed_v3_used"] is False
    assert manifest["writes_to_source_repos"] is False
    assert manifest["splits"]["train"]["rows"] == 1
    assert train_row["trace_construction_mode"] == "phase2t_dynamic_public_repo_repair_loop_trace"
    assert train_row["repair_loop_episode"]["stages"][-1]["stage"] == "emit_verified_stop"
    assert train_row["safety_controls"]["source_repo_read_only_observed"] is True
    assert train_row["normalization"]["phase2t_sealed_feedback_absent"] is True
    assert source_status == ""
    for rel_path in train_row["artifact_paths"].values():
        assert (tmp_path / "phase2t_traces" / rel_path).exists()
