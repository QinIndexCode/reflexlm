import json
import subprocess
from pathlib import Path

from reflexlm.cli.audit_phase2s_open_repair import (
    BASELINE_METHODS,
    build_phase2s_data_health,
    build_phase2s_full_holdout_postflight,
    build_phase2s_pretrain_gate,
    build_phase2s_smoke_postflight,
    compute_phase2s_baseline_predictions,
)
from reflexlm.cli.build_phase2s_head_dataset import (
    build_phase2s_head_dataset,
    phase2s_repair_trace_to_head_row,
)
from reflexlm.cli.build_phase2s_reproduction_report import (
    build_phase2s_reproduction_report,
)
from reflexlm.cli.build_phase2s_sealed_transfer_report import (
    build_phase2s_sealed_transfer_report,
)
from reflexlm.cli.collect_phase2s_public_repair_traces import (
    collect_phase2s_public_repair_traces,
)
from reflexlm.cli.generate_phase2s_open_repair_smoke import (
    build_phase2s_open_repair_smoke_dataset,
)
from reflexlm.cli.review_phase2s_public_repair_boundary import (
    build_phase2s_public_repair_boundary_review,
)


TASK_FAMILIES = [
    "dependency_or_import_mismatch",
    "localized_unit_assertion",
    "stale_snapshot_update",
    "config_or_environment_marker",
    "multi_file_traceback_relation",
]
EVIDENCE_DENSITIES = ["low", "medium", "high"]
CANDIDATE_COUNTS = [2, 3, 4]
REPAIR_DEPTHS = ["one_edit", "two_edits", "stale_state_refresh"]
FAILURE_OBSERVABILITY = [
    "direct_traceback",
    "indirect_changed_file_relation",
    "ambiguous_same_intent_command",
]
AMBIGUITY_CLASSES = ["same_intent_command", "same_file_read", "stage_transition"]
REPAIR_ACTIONS = [
    "repair_plan_alpha",
    "repair_plan_bravo",
    "repair_plan_charlie",
    "repair_plan_delta",
]


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _make_public_literal_repo(tmp_path: Path, repo_id: str) -> Path:
    repo = tmp_path / repo_id
    repo.mkdir(parents=True)
    (repo / "LICENSE").write_text("MIT fixture\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        "\n".join(
            [
                "[tool.pytest.ini_options]",
                'addopts = "--strict-config"',
                'unknown_phase2s_config_if_loaded = "must_be_isolated"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (repo / "module.py").write_text(
        "\n".join(
            [
                "def aardvark():",
                "    return (",
                "        'aa'",
                "        'bb'",
                "    )",
                "",
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
        ["git", "config", "user.email", "phase2s@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Phase2S Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    return repo


def _artifact_set(root: Path, split: str, repo_id: str) -> dict[str, str]:
    artifact_dir = root / "artifacts" / split / repo_id
    paths = {
        "patch_diff": artifact_dir / "patch.diff",
        "command_log": artifact_dir / "command_log.json",
        "test_output": artifact_dir / "test_output.json",
        "rollback_log": artifact_dir / "rollback_log.json",
        "sandbox_integrity_report": artifact_dir / "sandbox_integrity.json",
    }
    for key, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{key}: recorded\n", encoding="utf-8")
    return {key: path.relative_to(root).as_posix() for key, path in paths.items()}


def _repair_candidates(candidate_count: int) -> list[dict]:
    return [
        {
            "repair_action": action,
            "intent": "apply_patch_and_rerun_tests",
            "edit_scope": "bounded_source_patch",
            "description": "Apply a bounded source patch, rerun the failing test, and stop if verification passes.",
            "verification_command": "python -m pytest -q tests/test_repair_case.py --maxfail=1",
        }
        for action in REPAIR_ACTIONS[:candidate_count]
    ]


def _phase2s_rows(
    root: Path,
    split: str,
    *,
    count: int,
    source_kind: str = "synthetic_safe_repo",
    baseline_solved: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    slot_offsets = {candidate_count: 0 for candidate_count in CANDIDATE_COUNTS}
    for index in range(count):
        candidate_count = CANDIDATE_COUNTS[index % len(CANDIDATE_COUNTS)]
        expected_slot = slot_offsets[candidate_count] % candidate_count
        slot_offsets[candidate_count] += 1
        repo_id = f"{split}_repo_{index:03d}"
        candidates = _repair_candidates(candidate_count)
        expected_action = candidates[expected_slot]["repair_action"]
        row = {
            "trace_id": f"{split}:{repo_id}:{index}",
            "split": split,
            "source_kind": source_kind,
            "repo_id": repo_id,
            "repo_url_or_origin": (
                f"https://example.invalid/{repo_id}.git"
                if source_kind == "public_repo"
                else f"synthetic://phase2s/{repo_id}"
            ),
            "commit_hash": f"{index:040x}"[-40:],
            "license_or_synthetic_origin": "MIT synthetic-safe fixture",
            "collection_script_hash": "a" * 64,
            "normalization": {
                "deterministic": True,
                "redacted_absolute_local_paths": True,
                "redacted_secrets_tokens_and_emails": True,
                "preserved_runtime_visible_evidence": True,
            },
            "current_visible_text": (
                "Sandboxed repository repair task. Use failing test output, rollback "
                "evidence, and bounded edit scope to choose a repair action."
            ),
            "runtime_visible_evidence": {
                "failure_observability": FAILURE_OBSERVABILITY[
                    index % len(FAILURE_OBSERVABILITY)
                ],
                "failing_test_target": "tests/test_repair_case.py",
                "changed_files": ["repair_case.py"],
                "traceback_symbols": ["repair_case.evaluate"],
                "watched_files": ["tests/test_repair_case.py"],
                "prior_repair_summary": "Prior sandbox evidence was recorded without exposing the answer.",
                "stale_state_refresh": REPAIR_DEPTHS[index % len(REPAIR_DEPTHS)]
                == "stale_state_refresh",
                "source_repo_observed_read_only": True,
                "execution_sandbox_used": True,
            },
            "repair_candidates": candidates,
            "expected_repair_action": expected_action,
            "repair_runtime": {
                "patch_application_recorded": True,
                "post_patch_tests_recorded": True,
                "rollback_recorded": True,
                "sandbox_cleanup_recorded": True,
                "source_repo_read_only_observed": True,
                "bounded_edit_scope_observed": True,
                "command_allowlist_observed": True,
            },
            "artifact_paths": _artifact_set(root, split, repo_id),
            "difficulty": {
                "task_family": TASK_FAMILIES[index % len(TASK_FAMILIES)],
                "candidate_count": candidate_count,
                "evidence_density": EVIDENCE_DENSITIES[index % len(EVIDENCE_DENSITIES)],
                "repair_depth": REPAIR_DEPTHS[index % len(REPAIR_DEPTHS)],
                "failure_observability": FAILURE_OBSERVABILITY[
                    index % len(FAILURE_OBSERVABILITY)
                ],
                "ambiguity_class": AMBIGUITY_CLASSES[index % len(AMBIGUITY_CLASSES)],
            },
            "trace_hash": f"{split}-{repo_id}-{index}",
        }
        baselines = compute_phase2s_baseline_predictions(row)
        if baseline_solved:
            baselines = {name: expected_action for name in baselines}
        row["baselines"] = baselines
        row["baseline_metadata"] = {
            name: {
                "measured": True,
                "method": method,
                "uses_expected_repair_action": False,
                "uses_sealed_feedback": False,
            }
            for name, method in BASELINE_METHODS.items()
        }
        rows.append(row)
    return rows


def _write_splits(
    root: Path,
    *,
    source_kind: str = "synthetic_safe_repo",
    baseline_solved: bool = False,
) -> tuple[Path, Path, Path]:
    train = _write_jsonl(
        root / "train.raw.jsonl",
        _phase2s_rows(root, "train", count=15, source_kind=source_kind),
    )
    val = _write_jsonl(
        root / "val.raw.jsonl",
        _phase2s_rows(
            root,
            "val",
            count=15,
            source_kind=source_kind,
            baseline_solved=baseline_solved,
        ),
    )
    holdout = _write_jsonl(
        root / "holdout.raw.jsonl",
        _phase2s_rows(root, "holdout", count=9, source_kind=source_kind),
    )
    return train, val, holdout


def _phase2s_identity_rows(root: Path, split: str, *, count: int) -> list[dict]:
    rows = _phase2s_rows(root, split, count=count, source_kind="public_repo")
    for row_index, row in enumerate(rows):
        candidates = row["repair_candidates"]
        expected = row["expected_repair_action"]
        expected_slot = next(
            index for index, candidate in enumerate(candidates) if candidate["repair_action"] == expected
        )
        for candidate_index, candidate in enumerate(candidates):
            candidate["target_line"] = 40 + candidate_index
            candidate["target_col"] = 7 + candidate_index
            candidate["target_literal_hash"] = f"literal-{row_index}-{candidate_index}"
            candidate["structural_probe_hash"] = f"probe-{row_index}-{candidate_index}"
            if candidate_index == expected_slot:
                candidate["edit_scope"] = "repair_case.py"
                candidate["target_symbol"] = "repair_case.evaluate"
                row["runtime_visible_evidence"]["target_location"] = {
                    "path": "repair_case.py",
                    "line": candidate["target_line"],
                    "col": candidate["target_col"],
                }
                row["runtime_visible_evidence"]["expected_literal_hash"] = candidate[
                    "target_literal_hash"
                ]
                row["runtime_visible_evidence"]["structural_probe_hashes"] = [
                    candidate["structural_probe_hash"]
                ]
            else:
                candidate["edit_scope"] = f"distractor_{row_index}_{candidate_index}.py"
                candidate["target_symbol"] = f"distractor_{row_index}_{candidate_index}.evaluate"
        row["runtime_visible_evidence"]["changed_files"] = ["repair_case.py"]
        row["runtime_visible_evidence"]["traceback_symbols"] = ["repair_case.evaluate"]
    return rows


def test_phase2s_data_health_accepts_synthetic_smoke_but_blocks_claim_training(
    tmp_path: Path,
) -> None:
    train, val, holdout = _write_splits(tmp_path)

    report = build_phase2s_data_health(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
        dataset_root=tmp_path,
    )
    gate = build_phase2s_pretrain_gate(
        data_health_json=_write(tmp_path / "data_health.json", report)
    )

    assert report["passed"] is True
    assert report["claim_bearing_training_ready"] is False
    assert report["allowed_next_action"] == "collect_public_phase2s_repair_traces_before_training"
    assert gate["passed"] is False
    assert "do_not_train_phase2s_from_synthetic_smoke_only" in gate["blocked_actions"]


def test_phase2s_data_health_allows_public_claim_bearing_pretrain_gate(
    tmp_path: Path,
) -> None:
    train, val, holdout = _write_splits(tmp_path, source_kind="public_repo")

    report = build_phase2s_data_health(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
        dataset_root=tmp_path,
    )
    gate = build_phase2s_pretrain_gate(
        data_health_json=_write(tmp_path / "data_health.json", report)
    )

    assert report["passed"] is True
    assert report["claim_bearing_training_ready"] is True
    assert report["checks"]["phase2s_val_factor_level_coverage"] is True
    assert gate["passed"] is True
    assert gate["allowed_next_action"] == "run_phase2s_claim_bearing_smoke_training_only"


def test_phase2s_data_health_rejects_sealed_markers_and_candidate_slots(
    tmp_path: Path,
) -> None:
    train, val, holdout = _write_splits(tmp_path)
    rows = [json.loads(line) for line in val.read_text(encoding="utf-8").splitlines()]
    rows[0]["current_visible_text"] += " external_trace_v3_semantic_required candidate_0"
    val.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = build_phase2s_data_health(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
        dataset_root=tmp_path,
    )

    assert report["passed"] is False
    assert "do_not_use_sealed_or_sealed_failure_feedback" in report["blocked_actions"]
    assert "do_not_train_with_candidate_slot_markers_visible" in report["blocked_actions"]


def test_phase2s_data_health_rejects_baseline_solved_or_missing_artifacts(
    tmp_path: Path,
) -> None:
    train, val, holdout = _write_splits(tmp_path, baseline_solved=True)
    missing_artifact = tmp_path / "artifacts" / "val" / "val_repo_000" / "patch.diff"
    missing_artifact.unlink()

    report = build_phase2s_data_health(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
        dataset_root=tmp_path,
    )

    assert report["passed"] is False
    assert "do_not_train_when_any_required_baseline_solves_phase2s_val" in report[
        "blocked_actions"
    ]
    assert "do_not_train_without_patch_test_rollback_sandbox_artifacts" in report[
        "blocked_actions"
    ]


def test_phase2s_data_health_rejects_same_public_origin_across_splits(
    tmp_path: Path,
) -> None:
    train, val, holdout = _write_splits(tmp_path, source_kind="public_repo")
    train_rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    val_rows = [json.loads(line) for line in val.read_text(encoding="utf-8").splitlines()]
    val_rows[0]["repo_id"] = "val_repo_with_distinct_id_but_train_origin"
    val_rows[0]["repo_url_or_origin"] = train_rows[0]["repo_url_or_origin"]
    _write_jsonl(val, val_rows)

    report = build_phase2s_data_health(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
        dataset_root=tmp_path,
    )

    assert report["checks"]["phase2s_all_split_repos_disjoint"] is False
    assert "do_not_train_without_repo_disjoint_splits" in report["blocked_actions"]


def test_phase2s_smoke_generator_records_real_runtime_artifacts(tmp_path: Path) -> None:
    manifest = build_phase2s_open_repair_smoke_dataset(
        output_root=tmp_path / "generated",
        train_count=1,
        val_count=1,
        holdout_count=1,
        timeout_seconds=10,
    )
    train_row = json.loads(
        (tmp_path / "generated" / "train.raw.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    assert manifest["sealed_v3_used"] is False
    assert manifest["claim_bearing_training_evidence"] is False
    assert train_row["repair_runtime"]["post_patch_tests_recorded"] is True
    assert train_row["repair_runtime"]["rollback_recorded"] is True
    for rel_path in train_row["artifact_paths"].values():
        assert (tmp_path / "generated" / rel_path).exists()


def test_phase2s_public_repair_collector_keeps_source_repo_read_only(
    tmp_path: Path,
) -> None:
    source_repo = _make_public_literal_repo(tmp_path, "public_repo")
    specs = _write(
        tmp_path / "specs.json",
        [
            {
                "repo_id": "public_repo",
                "split": "train",
                "local_path": str(source_repo),
                "repo_url": "https://example.invalid/public_repo.git",
                "license": "MIT",
            },
            {
                "repo_id": "public_repo_val",
                "split": "val",
                "local_path": str(source_repo),
                "repo_url": "https://example.invalid/public_repo.git",
                "license": "MIT",
            },
            {
                "repo_id": "public_repo_holdout",
                "split": "holdout",
                "local_path": str(source_repo),
                "repo_url": "https://example.invalid/public_repo.git",
                "license": "MIT",
            },
        ],
    )

    manifest = collect_phase2s_public_repair_traces(
        repo_specs_json=specs,
        output_root=tmp_path / "public_traces",
        clone_root=tmp_path / "clones",
        rows_per_repo=1,
        timeout_seconds=10,
        no_clone=True,
        incremental_output=True,
    )
    row = json.loads(
        (tmp_path / "public_traces" / "train.raw.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )

    assert manifest["claim_bearing_training_candidate"] is True
    assert manifest["claim_bearing_training_evidence"] is True
    assert manifest["claim_bearing_training_ready"] is False
    assert manifest["claim_bearing_training_ready_requires"] == [
        "phase2s_open_repair_data_health",
        "phase2s_open_repair_pretrain_gate",
    ]
    assert manifest["writes_to_source_repos"] is False
    assert manifest["incremental_output_enabled"] is True
    assert (tmp_path / "public_traces" / "manifest.progress.json").exists()
    assert (tmp_path / "public_traces" / "holdout.raw.jsonl").exists()
    assert manifest["repos"][0]["source_repo_read_only_observed"] is True
    assert row["source_kind"] == "public_repo"
    assert row["repair_runtime"]["post_patch_tests_recorded"] is True
    assert row["repair_runtime"]["rollback_recorded"] is True
    assert row["repair_runtime"]["source_repo_read_only_observed"] is True
    assert row["trace_construction_mode"] == "public_repo_sandbox_literal_repair_trace"
    assert all(candidate.get("target_symbol") for candidate in row["repair_candidates"])
    assert all(candidate.get("target_literal_hash") for candidate in row["repair_candidates"])
    assert "expected_literal_hash" in row["runtime_visible_evidence"]
    assert row["runtime_visible_evidence"]["traceback_symbols"][0] in {
        candidate["target_symbol"] for candidate in row["repair_candidates"]
    }
    for rel_path in row["artifact_paths"].values():
        assert (tmp_path / "public_traces" / rel_path).exists()

    report = build_phase2s_data_health(
        train_jsonl=tmp_path / "public_traces" / "train.raw.jsonl",
        val_jsonl=tmp_path / "public_traces" / "val.raw.jsonl",
        holdout_jsonl=tmp_path / "public_traces" / "holdout.raw.jsonl",
        dataset_root=tmp_path / "public_traces",
        min_train_rows=1,
        min_val_rows=1,
        min_holdout_rows=1,
    )
    assert report["checks"]["phase2s_all_split_repos_disjoint"] is False


def test_phase2s_head_dataset_builder_records_public_repair_identity_signal(
    tmp_path: Path,
) -> None:
    train_rows = _phase2s_identity_rows(tmp_path, "train", count=3)
    val_rows = _phase2s_identity_rows(tmp_path, "val", count=3)
    train = _write_jsonl(tmp_path / "train.raw.jsonl", train_rows)
    val = _write_jsonl(tmp_path / "val.raw.jsonl", val_rows)
    data_health = _write(
        tmp_path / "data_health.json",
        {
            "passed": True,
            "effective_split_hashes": {
                "phase2s_train": "train_hash",
                "phase2s_val": "val_hash",
                "phase2s_holdout": "holdout_hash",
            },
        },
    )
    pretrain_gate = _write(tmp_path / "pretrain_gate.json", {"passed": True})

    manifest = build_phase2s_head_dataset(
        train_jsonl=train,
        val_jsonl=val,
        output_dir=tmp_path / "heads",
        data_health_json=data_health,
        pretrain_gate_json=pretrain_gate,
    )
    head_row = json.loads(
        (tmp_path / "heads" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    assert manifest["dataset_family"] == "phase2s_public_repair_head_dataset"
    assert manifest["json_text_target"] is False
    assert manifest["sealed_v3_used"] is False
    assert manifest["source_data_health_passed"] is True
    assert manifest["source_pretrain_gate_passed"] is True
    assert manifest["effective_split_hashes"]["phase2s_train"] == "train_hash"
    assert head_row["prompt_style"] == "phase2s_public_repair_head_v1"
    assert [
        command.split(" ", 1)[0] for command in head_row["candidate_commands"]
    ] == [candidate["repair_action"] for candidate in train_rows[0]["repair_candidates"]]
    assert head_row["candidate_commands"][head_row["command_slot"]].startswith(
        train_rows[0]["expected_repair_action"]
    )
    expected_candidate = train_rows[0]["repair_candidates"][head_row["command_slot"]]
    expected_command = head_row["candidate_commands"][head_row["command_slot"]]
    assert f"target_literal_hash={expected_candidate['target_literal_hash']}" in expected_command
    assert f"structural_probe_hash={expected_candidate['structural_probe_hash']}" in expected_command
    assert f"target_line={expected_candidate['target_line']}" in expected_command
    assert f"target_col={expected_candidate['target_col']}" in expected_command
    assert f"target_literal_hash={expected_candidate['target_literal_hash']}" in head_row["state_prompt"]
    assert head_row["patch_proposal_label"] == 1
    assert head_row["bounded_edit_scope_label"] == 1
    assert head_row["rollback_safety_label"] == 1
    assert head_row["test_selection_slot"] == 0
    assert head_row["stop_condition_label"] == 0
    assert head_row["progress_monitor_label"] == 1
    assert head_row["verification_state_label"] == 1
    assert head_row["source_trace"]["sealed_v3_used"] is False
    assert "phase2s_public_repair" in head_row["runtime_overrides"]
    assert "candidate_0" not in head_row["state_prompt"].lower()
    assert "slot_0" not in head_row["state_prompt"].lower()
    expected_slot = head_row["command_slot"]
    assert head_row["nsi_reference"][f"command_identity_slot:{expected_slot}"] > 0.0
    assert head_row["nsi_reference"]["command_identity_margin"] > 0.0


def test_phase2s_head_dataset_falls_back_when_data_health_hashes_missing(
    tmp_path: Path,
) -> None:
    train_rows = _phase2s_identity_rows(tmp_path, "train", count=1)
    val_rows = _phase2s_identity_rows(tmp_path, "val", count=1)
    train = _write_jsonl(tmp_path / "train.raw.jsonl", train_rows)
    val = _write_jsonl(tmp_path / "val.raw.jsonl", val_rows)
    data_health = _write(tmp_path / "data_health.json", {"passed": True})

    manifest = build_phase2s_head_dataset(
        train_jsonl=train,
        val_jsonl=val,
        output_dir=tmp_path / "heads",
        data_health_json=data_health,
    )

    assert manifest["source_data_health_passed"] is True
    assert len(manifest["effective_split_hashes"]["phase2s_train"]) == 64
    assert len(manifest["effective_split_hashes"]["phase2s_val"]) == 64


def test_phase2s_command_identity_uses_symbol_boundaries_not_substrings(
    tmp_path: Path,
) -> None:
    row = _phase2s_rows(tmp_path, "train", count=1, source_kind="public_repo")[0]
    candidates = row["repair_candidates"]
    row["expected_repair_action"] = candidates[0]["repair_action"]
    candidates[0]["edit_scope"] = "pkg/api.py"
    candidates[0]["target_symbol"] = "check"
    candidates[1]["edit_scope"] = "pkg/api.py"
    candidates[1]["target_symbol"] = "checkPath"
    row["runtime_visible_evidence"]["changed_files"] = ["pkg/api.py"]
    row["runtime_visible_evidence"]["traceback_symbols"] = ["check"]
    row["runtime_visible_evidence"].pop("expected_literal_hash", None)
    row["runtime_visible_evidence"].pop("target_location", None)

    head_row = phase2s_repair_trace_to_head_row(row)
    ref = head_row["nsi_reference"]

    assert ref["command_identity_slot:0"] > ref["command_identity_slot:1"]
    assert ref["command_identity_margin"] > 0.0


def test_phase2s_head_row_rejects_missing_expected_repair_action(tmp_path: Path) -> None:
    row = _phase2s_identity_rows(tmp_path, "train", count=1)[0]
    row["expected_repair_action"] = "missing_repair_action"

    try:
        phase2s_repair_trace_to_head_row(row)
    except ValueError as exc:
        assert "expected_repair_action not present" in str(exc)
    else:
        raise AssertionError("missing expected repair action should fail head-row conversion")


def test_phase2s_smoke_postflight_requires_model_delta_over_source_overlap(
    tmp_path: Path,
) -> None:
    data_health = _write(
        tmp_path / "data_health.json",
        {
            "passed": True,
            "effective_split_hashes": {
                "phase2s_train": "train_hash",
                "phase2s_val": "val_hash",
                "phase2s_holdout": "holdout_hash",
            },
        },
    )
    pretrain_gate = _write(
        tmp_path / "pretrain_gate.json",
        {
            "passed": True,
            "effective_split_hashes": {
                "phase2s_train": "train_hash",
                "phase2s_val": "val_hash",
                "phase2s_holdout": "holdout_hash",
            },
        },
    )
    head_manifest = _write(
        tmp_path / "head_manifest.json",
        {
            "source_data_health_passed": True,
            "source_pretrain_gate_passed": True,
            "effective_split_hashes": {
                "phase2s_train": "train_hash",
                "phase2s_val": "val_hash",
                "phase2s_holdout": "holdout_hash",
            },
        },
    )
    summary = {
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "use_pairwise_command_reranker": False,
        "command_candidate_encoder": "features_only",
        "source_overlap_command_slot_baseline": {"val": {"accuracy": 0.40}},
        "history": [{"val_metrics": {"command_slot_accuracy": 0.91}}],
        "run_manifest": {"duration_seconds": 120.0},
    }
    zero_nsi_diagnostics = {"sources": {"effective": {"accuracy": 0.50}}}
    accepted = build_phase2s_smoke_postflight(
        training_summary_json=_write(tmp_path / "summary.pass.json", summary),
        data_health_json=data_health,
        pretrain_gate_json=pretrain_gate,
        head_manifest_json=head_manifest,
        zero_nsi_diagnostics_json=_write(tmp_path / "zero_nsi.pass.json", zero_nsi_diagnostics),
        min_model_minus_zero_nsi=0.20,
    )
    summary["source_overlap_command_slot_baseline"] = {"val": {"accuracy": 0.91}}
    rejected = build_phase2s_smoke_postflight(
        training_summary_json=_write(tmp_path / "summary.fail.json", summary),
        data_health_json=data_health,
        pretrain_gate_json=pretrain_gate,
        head_manifest_json=head_manifest,
    )

    assert accepted["passed"] is True
    assert accepted["allowed_next_action"] == "run_phase2s_full_nonsealed_training_only"
    assert abs(accepted["metrics"]["model_minus_source_overlap_accuracy"] - 0.51) < 1e-9
    assert abs(accepted["metrics"]["model_minus_zero_nsi_accuracy"] - 0.41) < 1e-9
    assert rejected["passed"] is False
    assert "do_not_claim_phase2s_mechanism_delta_from_source_overlap" in rejected[
        "blocked_actions"
    ]

    summary["source_overlap_command_slot_baseline"] = {"val": {"accuracy": 0.40}}
    zero_nsi_diagnostics["sources"]["effective"]["accuracy"] = 0.85
    rejected_zero_nsi = build_phase2s_smoke_postflight(
        training_summary_json=_write(tmp_path / "summary.zero_nsi_fail.json", summary),
        data_health_json=data_health,
        pretrain_gate_json=pretrain_gate,
        head_manifest_json=head_manifest,
        zero_nsi_diagnostics_json=_write(
            tmp_path / "zero_nsi.fail.json", zero_nsi_diagnostics
        ),
        min_model_minus_zero_nsi=0.20,
    )
    assert rejected_zero_nsi["passed"] is False
    assert "do_not_claim_phase2s_nsi_identity_delta_from_smoke" in rejected_zero_nsi[
        "blocked_actions"
    ]


def test_phase2s_full_holdout_postflight_requires_repo_disjoint_holdout_delta(
    tmp_path: Path,
) -> None:
    hashes = {
        "phase2s_train": "train_hash",
        "phase2s_val": "val_hash",
        "phase2s_holdout": "holdout_hash",
    }
    data_health = _write(
        tmp_path / "data_health.json",
        {
            "passed": True,
            "effective_split_hashes": hashes,
            "rollups": {
                "holdout": {"rows": 9},
                "baselines": {
                    "holdout": {"source_overlap": {"accuracy": 0.35}},
                },
            },
        },
    )
    pretrain_gate = _write(
        tmp_path / "pretrain_gate.json",
        {"passed": True, "effective_split_hashes": hashes},
    )
    head_manifest = _write(
        tmp_path / "head_manifest.json",
        {
            "source_data_health_passed": True,
            "source_pretrain_gate_passed": True,
            "effective_split_hashes": hashes,
        },
    )
    summary = {
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "use_pairwise_command_reranker": False,
        "command_candidate_encoder": "features_only",
        "source_overlap_command_slot_baseline": {"val": {"accuracy": 0.35}},
        "history": [{"val_metrics": {"command_slot_accuracy": 0.96}}],
        "run_manifest": {"duration_seconds": 900.0},
    }
    holdout = {
        "sealed_data_used_for_training_or_tuning": False,
        "command_record_count": 9,
        "sources": {
            "effective": {"accuracy": 0.90},
            "source_overlap_baseline": {"accuracy": 0.35},
        },
    }
    zero_nsi = {"sources": {"effective": {"accuracy": 0.40}}}
    accepted = build_phase2s_full_holdout_postflight(
        training_summary_json=_write(tmp_path / "summary.json", summary),
        data_health_json=data_health,
        pretrain_gate_json=pretrain_gate,
        head_manifest_json=head_manifest,
        holdout_diagnostics_json=_write(tmp_path / "holdout.json", holdout),
        holdout_zero_nsi_diagnostics_json=_write(tmp_path / "zero_nsi.json", zero_nsi),
        min_holdout_model_minus_zero_nsi=0.20,
    )

    holdout["sources"]["effective"]["accuracy"] = 0.40
    rejected = build_phase2s_full_holdout_postflight(
        training_summary_json=_write(tmp_path / "summary.reject.json", summary),
        data_health_json=data_health,
        pretrain_gate_json=pretrain_gate,
        head_manifest_json=head_manifest,
        holdout_diagnostics_json=_write(tmp_path / "holdout.reject.json", holdout),
        holdout_zero_nsi_diagnostics_json=_write(tmp_path / "zero_nsi.reject.json", zero_nsi),
        min_holdout_model_minus_zero_nsi=0.20,
    )

    assert accepted["passed"] is True
    assert (
        accepted["allowed_next_action"]
        == "review_phase2s_boundary_before_package_or_sealed_final_eval"
    )
    assert abs(accepted["metrics"]["holdout_model_minus_source_overlap_accuracy"] - 0.55) < 1e-9
    assert rejected["passed"] is False
    assert "do_not_claim_phase2s_holdout_delta_from_source_overlap" in rejected[
        "blocked_actions"
    ]


def test_phase2s_boundary_review_keeps_strong_claims_bounded(tmp_path: Path) -> None:
    hashes = {
        "phase2s_train": "train_hash",
        "phase2s_val": "val_hash",
        "phase2s_holdout": "holdout_hash",
    }
    data_health = _write(
        tmp_path / "data_health.json",
        {
            "passed": True,
            "effective_split_hashes": hashes,
            "checks": {"phase2s_no_sealed_reference_anywhere": True},
            "rollups": {
                "train": {"rows": 768},
                "val": {"rows": 768},
                "holdout": {"rows": 896},
            },
        },
    )
    pretrain_gate = _write(
        tmp_path / "pretrain_gate.json",
        {"passed": True, "effective_split_hashes": hashes},
    )
    head_manifest = _write(
        tmp_path / "head_manifest.json",
        {
            "source_data_health_passed": True,
            "source_pretrain_gate_passed": True,
            "effective_split_hashes": hashes,
        },
    )
    smoke_postflight = _write(tmp_path / "smoke_postflight.json", {"passed": True})
    full_postflight = _write(
        tmp_path / "full_postflight.json",
        {"passed": True, "checks": {"no_json_motor_target": True}},
    )
    full_holdout = _write(
        tmp_path / "full_holdout.json",
        {
            "passed": True,
            "effective_split_hashes": hashes,
            "checks": {"holdout_diagnostics_not_sealed_tuned": True},
            "metrics": {
                "val_command_slot_accuracy": 0.98,
                "val_model_minus_source_overlap_accuracy": 0.60,
                "holdout_command_slot_accuracy": 0.91,
                "holdout_source_overlap_accuracy": 0.37,
                "holdout_zero_nsi_effective_accuracy": 0.36,
                "holdout_model_minus_source_overlap_accuracy": 0.54,
                "holdout_model_minus_zero_nsi_accuracy": 0.55,
                "use_pairwise_command_reranker": False,
                "low_level_qwen_calls_target": 0,
            },
        },
    )
    summary = _write(
        tmp_path / "summary.json",
        {
            "train_examples": 768,
            "config": {"max_train_records": 1024},
        },
    )

    report = build_phase2s_public_repair_boundary_review(
        data_health_json=data_health,
        pretrain_gate_json=pretrain_gate,
        head_manifest_json=head_manifest,
        smoke_postflight_json=smoke_postflight,
        full_postflight_json=full_postflight,
        full_holdout_postflight_json=full_holdout,
        full_training_summary_json=summary,
    )

    assert report["passed"] is True
    assert report["checks"]["all_available_public_train_rows_used"] is True
    assert report["metrics"]["public_train_supply_below_full_cap"] is True
    assert any("sealed cross-model transfer" in item for item in report["unsupported_claims"])
    assert "production autonomy" in report["claim_boundary"]


def test_phase2s_sealed_transfer_report_requires_boundary_and_records_debug_qwen(
    tmp_path: Path,
) -> None:
    def eval_json(
        name: str,
        *,
        completion: float,
        positives: int,
        model_calls: float,
        false_reflex: float,
        trace_rows: list[dict] | None = None,
    ) -> Path:
        run_path = tmp_path / "runs" / name
        run_path.mkdir(parents=True, exist_ok=True)
        if trace_rows is not None:
            (run_path / "trace_rows.jsonl").write_text(
                "\n".join(json.dumps(row) for row in trace_rows),
                encoding="utf-8",
            )
        return _write(
            tmp_path / f"{name}.json",
            {
                "episode_count": 10,
                "run_path": str(run_path),
                "metrics": {
                    "aggregate": {
                        "task_completion_rate": {
                            "mean": completion,
                            "positives": positives,
                        },
                        "model_calls": {"mean": model_calls},
                        "token_equivalent_cost": {"mean": model_calls * 256},
                        "reaction_latency_ms": {"mean": 10.0},
                        "state_hallucination_rate": {"mean": 0.0},
                        "false_reflex_rate": {"mean": false_reflex},
                    }
                },
            },
        )

    full = eval_json(
        "full",
        completion=1.0,
        positives=10,
        model_calls=1.0,
        false_reflex=0.0,
        trace_rows=[
            {
                "task_type": "test_failure_reflex",
                "qwen_called": True,
                "policy_debug": {"action_source": "native_head_cortex"},
            }
            for _ in range(10)
        ],
    )
    no_nsi = eval_json("no_nsi", completion=0.0, positives=0, model_calls=1.0, false_reflex=1.0)
    native = eval_json("native", completion=0.0, positives=0, model_calls=0.0, false_reflex=1.0)
    continuation = eval_json(
        "continuation",
        completion=0.0,
        positives=0,
        model_calls=0.0,
        false_reflex=1.0,
    )
    prompt = eval_json("prompt", completion=0.0, positives=0, model_calls=1.0, false_reflex=1.0)
    react = eval_json("react", completion=0.0, positives=0, model_calls=2.0, false_reflex=1.0)
    manifest = _write(
        tmp_path / "manifest.json",
        {
            "profile": "external_trace_v3_semantic_required",
            "sealed": True,
            "sealed_config_hash": "sealed_hash",
        },
    )
    boundary = _write(
        tmp_path / "boundary.json",
        {
            "passed": True,
            "allowed_next_action": "package_for_final_eval_only_after_manual_boundary_acceptance",
        },
    )

    report = build_phase2s_sealed_transfer_report(
        full_eval_json=full,
        prompt_eval_json=prompt,
        react_eval_json=react,
        no_nsi_eval_json=no_nsi,
        native_head_only_eval_json=native,
        continuation_only_eval_json=continuation,
        dataset_manifest_json=manifest,
        boundary_review_json=boundary,
    )
    missing_boundary = build_phase2s_sealed_transfer_report(
        full_eval_json=full,
        prompt_eval_json=prompt,
        react_eval_json=react,
        no_nsi_eval_json=no_nsi,
        native_head_only_eval_json=native,
        continuation_only_eval_json=continuation,
        dataset_manifest_json=manifest,
    )

    assert report["passed"] is True
    assert report["metrics"]["full_low_level_qwen_calls"] == 0
    assert report["metrics"]["full_debug_qwen_calls"] == 10
    assert report["metrics"]["full_model_calls"] == 1.0
    assert any("generation-free execution is not proven" in item for item in report["unsupported_claims"])
    assert missing_boundary["passed"] is False
    assert missing_boundary["checks"]["boundary_review_passed"] is False


def test_phase2s_reproduction_report_requires_distinct_models_and_bounded_claims(
    tmp_path: Path,
) -> None:
    hashes = {
        "phase2s_train": "train_hash",
        "phase2s_val": "val_hash",
        "phase2s_holdout": "holdout_hash",
    }

    def postflight(name: str, *, passed: bool = True, hashes_override: dict | None = None) -> Path:
        return _write(
            tmp_path / f"{name}.json",
            {
                "passed": passed,
                "effective_split_hashes": hashes_override or hashes,
                "checks": {
                    "pairwise_disabled_for_phase2s_full": True,
                    "low_level_qwen_calls_target_zero": True,
                    "no_json_motor_target": True,
                    "holdout_diagnostics_not_sealed_tuned": True,
                },
                "metrics": {
                    "val_command_slot_accuracy": 0.98,
                    "holdout_command_slot_accuracy": 0.91,
                    "holdout_model_minus_source_overlap_accuracy": 0.53,
                    "holdout_model_minus_zero_nsi_accuracy": 0.54,
                    "holdout_command_record_count": 896,
                    "duration_seconds": 600.0,
                },
            },
        )

    qwen3b = postflight("qwen3b")
    qwen7b = postflight("qwen7b")
    same_model = build_phase2s_reproduction_report(
        runs=[
            ("qwen2_5_3b", "13", qwen3b),
            ("qwen2_5_3b", "17", qwen7b),
        ],
    )
    cross_model = build_phase2s_reproduction_report(
        runs=[
            ("qwen2_5_3b", "13", qwen3b),
            ("qwen2_5_7b", "13", qwen7b),
        ],
    )
    multiseed_required = build_phase2s_reproduction_report(
        runs=[
            ("qwen2_5_3b", "13", qwen3b),
            ("qwen2_5_7b", "13", qwen7b),
        ],
        min_seeds_per_model=2,
    )
    multiseed_passed = build_phase2s_reproduction_report(
        runs=[
            ("qwen2_5_3b", "13", qwen3b),
            ("qwen2_5_3b", "17", qwen3b),
            ("qwen2_5_7b", "13", qwen7b),
            ("qwen2_5_7b", "17", qwen7b),
        ],
        min_seeds_per_model=2,
    )

    assert same_model["passed"] is False
    assert same_model["checks"]["distinct_model_count_minimum_met"] is False
    assert cross_model["passed"] is True
    assert "cross-model" in cross_model["supported_claims"][0]
    assert any("multi-seed robustness is not proven" in item for item in cross_model["unsupported_claims"])
    assert multiseed_required["passed"] is False
    assert multiseed_required["checks"]["seed_count_minimum_met"] is False
    assert multiseed_passed["passed"] is True
    assert any("multi-seed robustness is supported" in item for item in multiseed_passed["supported_claims"])
    assert not any(
        "multi-seed robustness is not proven" in item
        for item in multiseed_passed["unsupported_claims"]
    )
