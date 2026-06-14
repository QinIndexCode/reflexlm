import json
import subprocess
from pathlib import Path

from reflexlm.cli.audit_phase2aa_bounded_patch_candidate_execution import (
    audit_phase2aa_bounded_patch_candidate_execution,
)
from reflexlm.cli.audit_phase2aa_candidate_selection_delta_gate import (
    audit_phase2aa_candidate_selection_delta_gate,
)
from reflexlm.cli.audit_phase2aa_bounded_patch_candidates import (
    audit_phase2aa_bounded_patch_candidates,
)
import reflexlm.cli.run_phase2aa_bounded_patch_candidate_execution as phase2aa_runner
from reflexlm.cli.build_phase2aa_bounded_patch_candidates import (
    CLAIM_BOUNDARY,
    build_phase2aa_bounded_patch_candidates,
    phase2z_row_to_phase2aa,
)
from reflexlm.cli.build_phase2aa_candidate_selection_baseline_report import (
    build_phase2aa_candidate_selection_baseline_report,
)
from reflexlm.cli.run_phase2aa_bounded_patch_candidate_execution import (
    run_phase2aa_bounded_patch_candidate_execution,
)
from reflexlm.cli.run_phase2z_public_structural_repair_execution import _state_for_public_policy
from reflexlm.cli.run_phase2z_public_structural_repair_execution import _copy_public_repo
from reflexlm.llm.receptor_latent import runtime_command_identity_signal


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _execution_row(
    index: int,
    *,
    policy_loaded: bool,
    selected_slot: int,
    expected_slot: int,
    success: bool,
) -> dict:
    return {
        "trace_id": f"holdout:repo0:{index}",
        "source_kind": "public_repo",
        "claim_boundary": CLAIM_BOUNDARY,
        "policy_loaded": policy_loaded,
        "selected_patch_candidate_slot": selected_slot,
        "expected_patch_candidate_slot": expected_slot,
        "patch_candidate_selected_correctly": selected_slot == expected_slot,
        "success": success,
        "sealed_feedback_used": False,
        "claim_bearing_freeform_patch_evidence": False,
        "freeform_patch_generation": False,
    }


def _row(index: int, split: str = "val") -> dict:
    expected_slot = index % 2
    candidates = [
        {
            "repair_action": f"structural_repair_{index}_0",
            "intent": "apply_patch_and_rerun_tests",
            "structural_probe_hash": f"hash{index}0",
            "target_symbol": f"sym{index}0",
        },
        {
            "repair_action": f"structural_repair_{index}_1",
            "intent": "apply_patch_and_rerun_tests",
            "structural_probe_hash": f"hash{index}1",
            "target_symbol": f"sym{index}1",
        },
    ]
    expected = candidates[expected_slot]["repair_action"]
    return {
        "trace_id": f"{split}:repo0:{index}",
        "split": split,
        "source_kind": "public_repo",
        "repo_id": "repo0",
        "current_visible_text": "public runtime evidence without slot markers",
        "repair_candidates": candidates,
        "expected_repair_action": expected,
        "baselines": {
            "source_overlap": expected if index % 2 == 0 else candidates[1 - expected_slot]["repair_action"],
            "prompt_only": candidates[0]["repair_action"],
        },
        "normalization": {"sealed_feedback_absent": True},
    }


def test_phase2aa_builder_adds_bounded_patch_candidates_without_freeform_claim() -> None:
    converted = phase2z_row_to_phase2aa(_row(1))

    assert converted["claim_boundary"] == CLAIM_BOUNDARY
    assert converted["expected_patch_candidate_slot"] == 1
    assert converted["patch_candidates"][1]["patch_source"] == "recorded_correct_patch_artifact"
    assert converted["patch_candidates"][0]["freeform_patch_generation"] is False
    assert "free-form patch generation" in " ".join(converted["claim_boundary_notes"])


def test_phase2aa_data_health_accepts_nonzero_non_ceiling_controls(tmp_path: Path) -> None:
    train = _write_jsonl(tmp_path / "train.jsonl", [phase2z_row_to_phase2aa(_row(i, "train")) for i in range(24)])
    val = _write_jsonl(tmp_path / "val.jsonl", [phase2z_row_to_phase2aa(_row(i, "val")) for i in range(24)])
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", [phase2z_row_to_phase2aa(_row(i, "holdout")) for i in range(24)])

    report = audit_phase2aa_bounded_patch_candidates(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
    )

    assert report["passed"] is True
    assert report["metrics"]["best_non_full_baseline_accuracy"] == 0.5


def test_phase2aa_data_health_rejects_marker_leak(tmp_path: Path) -> None:
    rows = [phase2z_row_to_phase2aa(_row(i, "val")) for i in range(24)]
    rows[0]["current_visible_text"] = "gold candidate_0 leaked"
    train = _write_jsonl(tmp_path / "train.jsonl", [phase2z_row_to_phase2aa(_row(i, "train")) for i in range(24)])
    val = _write_jsonl(tmp_path / "val.jsonl", rows)
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", [phase2z_row_to_phase2aa(_row(i, "holdout")) for i in range(24)])

    report = audit_phase2aa_bounded_patch_candidates(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
    )

    assert report["passed"] is False
    assert report["checks"]["no_marker_leak_in_visible_text"] is False


def test_phase2aa_data_health_checks_runtime_artifact_availability(
    tmp_path: Path,
) -> None:
    rows = [phase2z_row_to_phase2aa(_row(i, "val")) for i in range(24)]
    for index, row in enumerate(rows):
        row["artifact_paths"] = {
            "patch_diff": f"artifacts/val/repo0/row_{index:05d}/patch.diff",
            "generated_test": f"artifacts/val/repo0/row_{index:05d}/generated_test.py",
        }
    train = _write_jsonl(tmp_path / "train.jsonl", rows)
    val = _write_jsonl(tmp_path / "val.jsonl", rows)
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", rows)

    missing_report = audit_phase2aa_bounded_patch_candidates(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
        artifact_root=tmp_path / "dataset",
    )

    assert missing_report["passed"] is False
    assert missing_report["checks"]["required_runtime_artifacts_available"] is False

    for index in range(24):
        artifact_dir = tmp_path / "dataset" / "artifacts" / "val" / "repo0" / f"row_{index:05d}"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "patch.diff").write_text("--- a/a.py\n+++ b/a.py\n", encoding="utf-8")
        (artifact_dir / "generated_test.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    present_report = audit_phase2aa_bounded_patch_candidates(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
        artifact_root=tmp_path / "dataset",
    )

    assert present_report["passed"] is True
    assert present_report["checks"]["required_runtime_artifacts_available"] is True


def test_phase2aa_candidate_selection_delta_gate_accepts_fixed_control_delta(
    tmp_path: Path,
) -> None:
    full_rows = [
        _execution_row(
            index,
            policy_loaded=True,
            selected_slot=index % 2,
            expected_slot=index % 2,
            success=True,
        )
        for index in range(24)
    ]
    control_rows = [
        _execution_row(
            index,
            policy_loaded=False,
            selected_slot=0,
            expected_slot=index % 2,
            success=(index % 2 == 0),
        )
        for index in range(24)
    ]

    report = audit_phase2aa_candidate_selection_delta_gate(
        full_execution_jsonl=_write_jsonl(tmp_path / "full.jsonl", full_rows),
        control_execution_jsonl=_write_jsonl(tmp_path / "control.jsonl", control_rows),
    )

    assert report["passed"] is True
    assert report["checks"]["control_is_fixed_non_oracle_slot"] is True
    assert report["metrics"]["full_minus_control_selection_accuracy"] == 0.5
    assert (
        "phase2aa_bounded_candidate_selection_package_delta_supported"
        in report["supported_claims"]
    )


def test_phase2aa_candidate_selection_delta_gate_rejects_oracle_control(
    tmp_path: Path,
) -> None:
    full_rows = [
        _execution_row(
            index,
            policy_loaded=True,
            selected_slot=index % 2,
            expected_slot=index % 2,
            success=True,
        )
        for index in range(24)
    ]
    oracle_control_rows = [
        _execution_row(
            index,
            policy_loaded=False,
            selected_slot=index % 2,
            expected_slot=index % 2,
            success=True,
        )
        for index in range(24)
    ]

    report = audit_phase2aa_candidate_selection_delta_gate(
        full_execution_jsonl=_write_jsonl(tmp_path / "full.jsonl", full_rows),
        control_execution_jsonl=_write_jsonl(tmp_path / "control.jsonl", oracle_control_rows),
    )

    assert report["passed"] is False
    assert report["checks"]["control_is_fixed_non_oracle_slot"] is False
    assert report["metrics"]["full_minus_control_selection_accuracy"] == 0.0
    assert "do_not_claim_phase2aa_candidate_selection_delta" in report["blocked_actions"]


def test_phase2aa_candidate_selection_delta_gate_rejects_trace_mismatch(
    tmp_path: Path,
) -> None:
    full_rows = [
        _execution_row(
            index,
            policy_loaded=True,
            selected_slot=index % 2,
            expected_slot=index % 2,
            success=True,
        )
        for index in range(24)
    ]
    control_rows = [
        _execution_row(
            index,
            policy_loaded=False,
            selected_slot=0,
            expected_slot=index % 2,
            success=(index % 2 == 0),
        )
        for index in range(24)
    ]
    control_rows[0]["trace_id"] = "holdout:repo1:different"

    report = audit_phase2aa_candidate_selection_delta_gate(
        full_execution_jsonl=_write_jsonl(tmp_path / "full.jsonl", full_rows),
        control_execution_jsonl=_write_jsonl(tmp_path / "control.jsonl", control_rows),
    )

    assert report["passed"] is False
    assert report["checks"]["same_trace_order"] is False


def test_phase2aa_build_writes_split_manifest(tmp_path: Path) -> None:
    source_train = _write_jsonl(tmp_path / "src_train.jsonl", [_row(i, "train") for i in range(2)])
    source_val = _write_jsonl(tmp_path / "src_val.jsonl", [_row(i, "val") for i in range(2)])
    source_holdout = _write_jsonl(tmp_path / "src_holdout.jsonl", [_row(i, "holdout") for i in range(2)])

    manifest = build_phase2aa_bounded_patch_candidates(
        train_jsonl=source_train,
        val_jsonl=source_val,
        holdout_jsonl=source_holdout,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
    )

    assert manifest["split_counts"] == {"train": 2, "val": 2, "holdout": 2}
    assert (tmp_path / "out" / "val.jsonl").exists()
    assert manifest["freeform_patch_generation"] is False


def _make_public_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "clones" / "repo0"
    repo.mkdir(parents=True)
    (repo / "a.py").write_text("import os\n\nVALUE = os.name\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "phase2aa@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Phase2AA Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    return repo


def test_public_repo_copy_accepts_behavioral_repo_alias(tmp_path: Path) -> None:
    clone_root = tmp_path / "clones"
    source_repo = clone_root / "repo0"
    source_repo.mkdir(parents=True)
    (source_repo / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
    sandbox = tmp_path / "sandbox"

    copied_from = _copy_public_repo(
        {"repo_id": "repo0_behavioral"},
        clone_root,
        sandbox,
    )

    assert copied_from == source_repo
    assert (sandbox / "a.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_phase2aa_runner_uses_selected_patch_candidate_slot(tmp_path: Path) -> None:
    repo = _make_public_repo(tmp_path)
    shadow_pkg = repo / "pytest"
    shadow_pkg.mkdir()
    (shadow_pkg / "__init__.py").write_text(
        "raise ModuleNotFoundError('repo-local pytest shadow should not be imported')\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True)
    patch = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,2 +1,3 @@\n"
        "+import os\n"
        " \n"
        " VALUE = os.name\n"
    )
    (artifact_dir / "patch.diff").write_text(patch, encoding="utf-8")
    (artifact_dir / "generated_test.py").write_text(
        "from pathlib import Path\n\n"
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n\n"
        "def test_import_restored():\n"
        "    assert 'import os' in (REPO_ROOT / 'a.py').read_text(encoding='utf-8')\n",
        encoding="utf-8",
    )
    row = phase2z_row_to_phase2aa(
        {
            "trace_id": "holdout:repo0:0",
            "split": "holdout",
            "source_kind": "public_repo",
            "repo_id": "repo0",
            "repo_url_or_origin": "https://example.invalid/repo0.git",
            "commit_hash": "a" * 40,
            "current_visible_text": "public runtime evidence",
            "runtime_visible_evidence": {"changed_files": ["a.py"]},
            "repair_candidates": [
                {"repair_action": "wrong", "verification_command": "pytest"},
                {"repair_action": "correct", "verification_command": "pytest"},
            ],
            "expected_repair_action": "correct",
            "expected_repair_result": {"test_target": "phase2z_repair_tests/test_case.py"},
            "artifact_paths": {
                "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
            },
        }
    )
    rows = _write_jsonl(tmp_path / "rows.jsonl", [row])
    package = _write_json(
        tmp_path / "package" / "native_nervous_package.json",
        {
            "policy_label": "test-policy",
            "open_repair_capabilities": {
                "patch_proposal": True,
                "bounded_edit_scope": True,
                "test_selection": True,
                "rollback_safety": True,
                "progress_monitor": True,
                "verification_state": True,
                "stop_condition": True,
            },
        },
    )

    report = run_phase2aa_bounded_patch_candidate_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
    )

    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert report["successes"] == 1
    assert result["selected_patch_candidate_slot"] == 1
    assert result["generated_test_source"] == "recorded_generated_test_direct"
    assert result["selected_tests"] == ["python phase2z_repair_tests/test_case.py"]
    assert result["patch_candidate_selected_correctly"] is True
    assert result["claim_bearing_candidate_selection_evidence"] is True
    assert result["claim_bearing_freeform_patch_evidence"] is False


def test_phase2aa_runner_accepts_raw_repair_candidate_rows(tmp_path: Path) -> None:
    _make_public_repo(tmp_path)
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True)
    patch = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,2 +1,3 @@\n"
        "+import os\n"
        " \n"
        " VALUE = os.name\n"
    )
    (artifact_dir / "patch.diff").write_text(patch, encoding="utf-8")
    (artifact_dir / "generated_test.py").write_text(
        "from pathlib import Path\n\n"
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n\n"
        "def test_import_restored():\n"
        "    assert 'import os' in (REPO_ROOT / 'a.py').read_text(encoding='utf-8')\n",
        encoding="utf-8",
    )
    raw_row = {
        "trace_id": "holdout:repo0:raw",
        "split": "holdout",
        "source_kind": "public_repo",
        "repo_id": "repo0",
        "repo_url_or_origin": "https://example.invalid/repo0.git",
        "commit_hash": "a" * 40,
        "current_visible_text": "public runtime evidence",
        "runtime_visible_evidence": {"changed_files": ["a.py"]},
        "repair_candidates": [
            {"repair_action": "wrong", "verification_command": "pytest"},
            {"repair_action": "correct", "verification_command": "pytest"},
        ],
        "expected_repair_action": "correct",
        "expected_repair_result": {"test_target": "phase2z_repair_tests/test_case.py"},
        "artifact_paths": {
            "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
            "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
        },
    }
    rows = _write_jsonl(tmp_path / "raw_rows.jsonl", [raw_row])
    package = _write_json(
        tmp_path / "package" / "native_nervous_package.json",
        {
            "policy_label": "raw-row-policy",
            "open_repair_capabilities": {
                "patch_proposal": True,
                "bounded_edit_scope": True,
                "test_selection": True,
                "rollback_safety": True,
                "progress_monitor": True,
                "verification_state": True,
                "stop_condition": True,
            },
        },
    )

    report = run_phase2aa_bounded_patch_candidate_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
    )

    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert report["successes"] == 1
    assert result["expected_patch_candidate_slot"] == 1
    assert result["selected_patch_candidate_slot"] == 1


def test_phase2aa_direct_generated_test_can_import_repo_local_package(tmp_path: Path) -> None:
    repo = _make_public_repo(tmp_path)
    package_dir = repo / "repo0pkg"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("VALUE = 'repo-local'\n", encoding="utf-8")
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True)
    patch = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,2 +1,3 @@\n"
        "+import os\n"
        " \n"
        " VALUE = os.name\n"
    )
    (artifact_dir / "patch.diff").write_text(patch, encoding="utf-8")
    (artifact_dir / "generated_test.py").write_text(
        "import repo0pkg\n"
        "from pathlib import Path\n\n"
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n\n"
        "def test_import_restored():\n"
        "    assert repo0pkg.VALUE == 'repo-local'\n"
        "    assert 'import os' in (REPO_ROOT / 'a.py').read_text(encoding='utf-8')\n",
        encoding="utf-8",
    )
    row = phase2z_row_to_phase2aa(
        {
            "trace_id": "holdout:repo0:repo-local-import",
            "split": "holdout",
            "source_kind": "public_repo",
            "repo_id": "repo0",
            "repo_url_or_origin": "https://example.invalid/repo0.git",
            "commit_hash": "a" * 40,
            "current_visible_text": "public runtime evidence",
            "runtime_visible_evidence": {"changed_files": ["a.py"]},
            "repair_candidates": [
                {"repair_action": "wrong", "verification_command": "pytest"},
                {"repair_action": "correct", "verification_command": "pytest"},
            ],
            "expected_repair_action": "correct",
            "expected_repair_result": {"test_target": "phase2z_repair_tests/test_case.py"},
            "artifact_paths": {
                "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
            },
        }
    )
    rows = _write_jsonl(tmp_path / "rows.jsonl", [row])
    package = _write_json(
        tmp_path / "package" / "native_nervous_package.json",
        {
            "policy_label": "repo-local-import-policy",
            "open_repair_capabilities": {
                "patch_proposal": True,
                "bounded_edit_scope": True,
                "test_selection": True,
                "rollback_safety": True,
                "progress_monitor": True,
                "verification_state": True,
                "stop_condition": True,
            },
        },
    )

    report = run_phase2aa_bounded_patch_candidate_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
    )

    assert report["successes"] == 1


def test_phase2aa_generated_test_loading_tests_tree_uses_pytest_runner(
    tmp_path: Path,
) -> None:
    repo = _make_public_repo(tmp_path)
    tests_dir = repo / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "conftest.py").write_text(
        "import pytest\nRESTORED = True\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "tests tree"], cwd=repo, check=True, capture_output=True)
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True)
    patch = (
        "--- a/tests/conftest.py\n"
        "+++ b/tests/conftest.py\n"
        "@@ -1 +1,2 @@\n"
        " import pytest\n"
        "+RESTORED = True\n"
    )
    (artifact_dir / "patch.diff").write_text(patch, encoding="utf-8")
    (artifact_dir / "generated_test.py").write_text(
        "import importlib.util\n"
        "from pathlib import Path\n\n"
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n\n"
        "def _load_module(name, path):\n"
        "    spec = importlib.util.spec_from_file_location(name, path)\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    assert spec is not None and spec.loader is not None\n"
        "    spec.loader.exec_module(module)\n"
        "    return module\n\n"
        "def test_conftest_restored():\n"
        "    module = _load_module('loaded_conftest', (REPO_ROOT / 'tests/conftest.py'))\n"
        "    assert module.RESTORED is True\n",
        encoding="utf-8",
    )
    row = phase2z_row_to_phase2aa(
        {
            "trace_id": "holdout:repo0:tests-tree",
            "split": "holdout",
            "source_kind": "public_repo",
            "repo_id": "repo0",
            "repo_url_or_origin": "https://example.invalid/repo0.git",
            "commit_hash": "a" * 40,
            "current_visible_text": "public runtime evidence",
            "runtime_visible_evidence": {"changed_files": ["tests/conftest.py"]},
            "repair_candidates": [
                {"repair_action": "wrong", "verification_command": "pytest"},
                {"repair_action": "correct", "verification_command": "pytest"},
            ],
            "expected_repair_action": "correct",
            "expected_repair_result": {"test_target": "phase2z_repair_tests/test_case.py"},
            "artifact_paths": {
                "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
            },
        }
    )
    rows = _write_jsonl(tmp_path / "rows.jsonl", [row])
    package = _write_json(
        tmp_path / "package" / "native_nervous_package.json",
        {
            "policy_label": "tests-tree-policy",
            "open_repair_capabilities": {
                "patch_proposal": True,
                "bounded_edit_scope": True,
                "test_selection": True,
                "rollback_safety": True,
                "progress_monitor": True,
                "verification_state": True,
                "stop_condition": True,
            },
        },
    )

    report = run_phase2aa_bounded_patch_candidate_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
    )

    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert report["successes"] == 1
    assert result["generated_test_source"] == "recorded_generated_test_pytest"
    assert result["selected_tests"] == ["python -m pytest -q phase2z_repair_tests/test_case.py --maxfail=1"]


def test_phase2aa_runner_records_learned_descriptor_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = _make_public_repo(tmp_path)
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "patch.diff").write_text(
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,2 +1,3 @@\n"
        "+import os\n"
        " \n"
        " VALUE = os.name\n",
        encoding="utf-8",
    )
    (artifact_dir / "generated_test.py").write_text(
        "from pathlib import Path\n\n"
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n\n"
        "def test_import_restored():\n"
        "    assert 'import os' in (REPO_ROOT / 'a.py').read_text(encoding='utf-8')\n",
        encoding="utf-8",
    )
    row = phase2z_row_to_phase2aa(
        {
            "trace_id": "holdout:repo0:descriptor-output",
            "split": "holdout",
            "source_kind": "public_repo",
            "repo_id": repo.name,
            "repo_url_or_origin": "https://example.invalid/repo0.git",
            "commit_hash": "a" * 40,
            "current_visible_text": "public runtime evidence",
            "runtime_visible_evidence": {
                "changed_files": ["a.py"],
                "structural_probe_hashes": ["correct-hash"],
            },
            "repair_candidates": [
                {
                    "repair_action": "correct",
                    "structural_probe_hash": "correct-hash",
                    "verification_command": "pytest",
                },
                {
                    "repair_action": "wrong",
                    "structural_probe_hash": "wrong-hash",
                    "verification_command": "pytest",
                },
            ],
            "expected_repair_action": "correct",
            "expected_repair_result": {"test_target": "phase2z_repair_tests/test_case.py"},
            "artifact_paths": {
                "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
            },
        }
    )
    rows = _write_jsonl(tmp_path / "rows.jsonl", [row])
    package = _write_json(
        tmp_path / "package" / "native_nervous_package.json",
        {"policy_label": "phase2au-descriptor-policy", "open_repair_capabilities": {}},
    )

    class FakePolicy:
        def __init__(self, _package):
            self.last_call = {}

        def act(self, _state):
            self.last_call = {
                "cortex_plan": {"command_slot": 0},
                "open_repair_head_outputs": {
                    "patch_proposal": 1,
                    "bounded_edit_scope": 1,
                    "rollback_safety": 1,
                    "test_selection_slot": 0,
                    "progress_monitor": 1,
                    "verification_state": 1,
                    "stop_condition": 0,
                },
                "learned_patch_descriptor_outputs": {
                    "patch_operation_index": 2,
                    "patch_operation": "insert_import",
                    "patch_target_file_slot": 0,
                    "patch_template_slot": 1,
                    "patch_template": "import_restoration",
                },
            }

    monkeypatch.setattr(phase2aa_runner, "NativeNervousPolicyPackage", FakePolicy)

    report = phase2aa_runner.run_phase2aa_bounded_patch_candidate_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
    )

    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    transcript = json.loads(
        Path(result["artifact_paths"]["transcript"]).read_text(encoding="utf-8")
    )
    assert report["successes"] == 1
    assert result["policy_learned_patch_descriptor_outputs"][
        "patch_operation"
    ] == "insert_import"
    assert transcript["learned_patch_descriptor_outputs"]["patch_template"] == (
        "import_restoration"
    )


def test_phase2aa_runner_derives_patch_observable_test_when_generated_test_missing(
    tmp_path: Path,
) -> None:
    _make_public_repo(tmp_path)
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True)
    patch = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,2 +1,3 @@\n"
        "+import os\n"
        " \n"
        " VALUE = os.name\n"
    )
    (artifact_dir / "patch.diff").write_text(patch, encoding="utf-8")
    row = phase2z_row_to_phase2aa(
        {
            "trace_id": "holdout:repo0:fallback-test",
            "split": "holdout",
            "source_kind": "public_repo",
            "repo_id": "repo0",
            "repo_url_or_origin": "https://example.invalid/repo0.git",
            "commit_hash": "a" * 40,
            "current_visible_text": "public runtime evidence",
            "runtime_visible_evidence": {"changed_files": ["a.py"]},
            "repair_candidates": [
                {"repair_action": "wrong", "verification_command": "pytest"},
                {"repair_action": "correct", "verification_command": "pytest"},
            ],
            "expected_repair_action": "correct",
            "expected_repair_result": {"test_target": "phase2s_repair_tests/test_case.py"},
            "artifact_paths": {
                "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
            },
        }
    )
    rows = _write_jsonl(tmp_path / "rows.jsonl", [row])
    package = _write_json(
        tmp_path / "package" / "native_nervous_package.json",
        {
            "policy_label": "fallback-test-policy",
            "open_repair_capabilities": {
                "patch_proposal": True,
                "bounded_edit_scope": True,
                "test_selection": True,
                "rollback_safety": True,
                "progress_monitor": True,
                "verification_state": True,
                "stop_condition": True,
            },
        },
    )

    report = run_phase2aa_bounded_patch_candidate_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
    )

    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert report["successes"] == 1
    assert result["generated_test_source"] == "patch_observable_generated_test"
    assert result["selected_patch_candidate_slot"] == 1


def test_phase2aa_patch_observable_test_rejects_distractor_marker_patch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _make_public_repo(tmp_path)
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True)
    patch = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,2 +1,3 @@\n"
        "+import os\n"
        " \n"
        " VALUE = os.name\n"
    )
    (artifact_dir / "patch.diff").write_text(patch, encoding="utf-8")
    row = phase2z_row_to_phase2aa(
        {
            "trace_id": "holdout:repo0:fallback-distractor",
            "split": "holdout",
            "source_kind": "public_repo",
            "repo_id": "repo0",
            "repo_url_or_origin": "https://example.invalid/repo0.git",
            "commit_hash": "a" * 40,
            "current_visible_text": "public runtime evidence",
            "runtime_visible_evidence": {"changed_files": ["a.py"]},
            "repair_candidates": [
                {"repair_action": "wrong", "verification_command": "pytest"},
                {"repair_action": "correct", "verification_command": "pytest"},
            ],
            "expected_repair_action": "correct",
            "expected_repair_result": {"test_target": "phase2s_repair_tests/test_case.py"},
            "artifact_paths": {
                "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
            },
        }
    )
    rows = _write_jsonl(tmp_path / "rows.jsonl", [row])
    package = _write_json(
        tmp_path / "package" / "native_nervous_package.json",
        {"policy_label": "fallback-distractor-policy", "open_repair_capabilities": {}},
    )

    class FakePolicy:
        def __init__(self, _package):
            self.last_call = {}

        def act(self, _state):
            self.last_call = {
                "cortex_plan": {"command_slot": 0},
                "open_repair_head_outputs": {
                    "patch_proposal": 1,
                    "bounded_edit_scope": 1,
                    "rollback_safety": 1,
                    "test_selection_slot": 0,
                    "progress_monitor": 1,
                    "verification_state": 1,
                    "stop_condition": 0,
                },
            }

    monkeypatch.setattr(phase2aa_runner, "NativeNervousPolicyPackage", FakePolicy)

    report = phase2aa_runner.run_phase2aa_bounded_patch_candidate_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
    )

    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert report["successes"] == 0
    assert result["selected_patch_candidate_slot"] == 0
    assert result["patch_candidate_selected_correctly"] is False
    assert result["full_test_pass_rate"] == 0.0


def test_phase2aa_runner_can_retry_bounded_candidates_after_verification_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _make_public_repo(tmp_path)
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True)
    patch = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,2 +1,3 @@\n"
        "+import os\n"
        " \n"
        " VALUE = os.name\n"
    )
    (artifact_dir / "patch.diff").write_text(patch, encoding="utf-8")
    (artifact_dir / "generated_test.py").write_text(
        "from pathlib import Path\n\n"
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n\n"
        "def test_import_restored():\n"
        "    assert 'import os' in (REPO_ROOT / 'a.py').read_text(encoding='utf-8')\n",
        encoding="utf-8",
    )
    row = phase2z_row_to_phase2aa(
        {
            "trace_id": "holdout:repo0:retry",
            "split": "holdout",
            "source_kind": "public_repo",
            "repo_id": "repo0",
            "repo_url_or_origin": "https://example.invalid/repo0.git",
            "commit_hash": "a" * 40,
            "current_visible_text": "public runtime evidence",
            "runtime_visible_evidence": {"changed_files": ["a.py"]},
            "repair_candidates": [
                {"repair_action": "wrong", "verification_command": "pytest"},
                {"repair_action": "correct", "verification_command": "pytest"},
            ],
            "expected_repair_action": "correct",
            "expected_repair_result": {"test_target": "phase2z_repair_tests/test_case.py"},
            "artifact_paths": {
                "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
            },
        }
    )
    rows = _write_jsonl(tmp_path / "rows.jsonl", [row])
    package = _write_json(
        tmp_path / "package" / "native_nervous_package.json",
        {"policy_label": "retry-policy", "open_repair_capabilities": {}},
    )

    class FakePolicy:
        def __init__(self, _package):
            self.last_call = {}

        def act(self, _state):
            self.last_call = {
                "cortex_plan": {"command_slot": 0},
                "open_repair_head_outputs": {
                    "patch_proposal": 1,
                    "bounded_edit_scope": 1,
                    "rollback_safety": 1,
                    "test_selection_slot": 0,
                    "progress_monitor": 1,
                    "verification_state": 1,
                    "stop_condition": 0,
                },
            }

    monkeypatch.setattr(phase2aa_runner, "NativeNervousPolicyPackage", FakePolicy)

    report = phase2aa_runner.run_phase2aa_bounded_patch_candidate_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        allow_bounded_candidate_retry=True,
    )

    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert report["successes"] == 1
    assert result["initial_selected_patch_candidate_slot"] == 0
    assert result["selected_patch_candidate_slot"] == 1
    assert result["candidate_attempts"][0]["passed"] is False
    assert result["candidate_attempts"][1]["passed"] is True


def test_phase2aa_policyless_retry_can_start_from_fixed_baseline_slot(tmp_path: Path) -> None:
    _make_public_repo(tmp_path)
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True)
    patch = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,2 +1,3 @@\n"
        "+import os\n"
        " \n"
        " VALUE = os.name\n"
    )
    (artifact_dir / "patch.diff").write_text(patch, encoding="utf-8")
    (artifact_dir / "generated_test.py").write_text(
        "from pathlib import Path\n\n"
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n\n"
        "def test_import_restored():\n"
        "    assert 'import os' in (REPO_ROOT / 'a.py').read_text(encoding='utf-8')\n",
        encoding="utf-8",
    )
    row = phase2z_row_to_phase2aa(
        {
            "trace_id": "holdout:repo0:policyless-retry",
            "split": "holdout",
            "source_kind": "public_repo",
            "repo_id": "repo0",
            "repo_url_or_origin": "https://example.invalid/repo0.git",
            "commit_hash": "a" * 40,
            "current_visible_text": "public runtime evidence",
            "runtime_visible_evidence": {"changed_files": ["a.py"]},
            "repair_candidates": [
                {"repair_action": "wrong", "verification_command": "pytest"},
                {"repair_action": "correct", "verification_command": "pytest"},
            ],
            "expected_repair_action": "correct",
            "expected_repair_result": {"test_target": "phase2z_repair_tests/test_case.py"},
            "artifact_paths": {
                "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
            },
        }
    )
    rows = _write_jsonl(tmp_path / "rows.jsonl", [row])
    package = _write_json(
        tmp_path / "package" / "native_nervous_package.json",
        {
            "policy_label": "policyless-retry",
            "open_repair_capabilities": {
                "patch_proposal": True,
                "bounded_edit_scope": True,
                "test_selection": True,
                "rollback_safety": True,
                "progress_monitor": True,
                "verification_state": True,
                "stop_condition": True,
            },
        },
    )

    report = run_phase2aa_bounded_patch_candidate_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
        allow_bounded_candidate_retry=True,
        policyless_start_slot=0,
    )

    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert report["successes"] == 1
    assert result["initial_selected_patch_candidate_slot"] == 0
    assert result["selected_patch_candidate_slot"] == 1
    assert len(result["candidate_attempts"]) == 2


def test_phase2aa_identity_first_retry_prioritizes_visible_identity_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _make_public_repo(tmp_path)
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True)
    patch = (
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,2 +1,3 @@\n"
        "+import os\n"
        " \n"
        " VALUE = os.name\n"
    )
    (artifact_dir / "patch.diff").write_text(patch, encoding="utf-8")
    (artifact_dir / "generated_test.py").write_text(
        "from pathlib import Path\n\n"
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n\n"
        "def test_import_restored():\n"
        "    assert 'import os' in (REPO_ROOT / 'a.py').read_text(encoding='utf-8')\n",
        encoding="utf-8",
    )
    row = phase2z_row_to_phase2aa(
        {
            "trace_id": "holdout:repo0:identity-first-retry",
            "split": "holdout",
            "source_kind": "public_repo",
            "repo_id": "repo0",
            "repo_url_or_origin": "https://example.invalid/repo0.git",
            "commit_hash": "a" * 40,
            "current_visible_text": "public runtime evidence",
            "runtime_visible_evidence": {
                "changed_files": ["a.py"],
                "structural_probe_hashes": ["correct-hash"],
            },
            "repair_candidates": [
                {"repair_action": "wrong0", "structural_probe_hash": "wrong0", "verification_command": "pytest"},
                {"repair_action": "wrong1", "structural_probe_hash": "wrong1", "verification_command": "pytest"},
                {
                    "repair_action": "correct",
                    "structural_probe_hash": "correct-hash",
                    "verification_command": "pytest",
                },
            ],
            "expected_repair_action": "correct",
            "expected_repair_result": {"test_target": "phase2z_repair_tests/test_case.py"},
            "artifact_paths": {
                "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
            },
        }
    )
    rows = _write_jsonl(tmp_path / "rows.jsonl", [row])
    package = _write_json(
        tmp_path / "package" / "native_nervous_package.json",
        {"policy_label": "identity-first-retry-policy", "open_repair_capabilities": {}},
    )

    class FakePolicy:
        def __init__(self, _package):
            self.last_call = {}

        def act(self, _state):
            self.last_call = {
                "cortex_plan": {"command_slot": 0},
                "open_repair_head_outputs": {
                    "patch_proposal": 1,
                    "bounded_edit_scope": 1,
                    "rollback_safety": 1,
                    "test_selection_slot": 0,
                    "progress_monitor": 1,
                    "verification_state": 1,
                    "stop_condition": 0,
                },
            }

    monkeypatch.setattr(phase2aa_runner, "NativeNervousPolicyPackage", FakePolicy)

    report = phase2aa_runner.run_phase2aa_bounded_patch_candidate_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        allow_bounded_candidate_retry=True,
        max_candidate_attempts=2,
        retry_prioritization="identity_first",
    )

    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert report["successes"] == 1
    assert result["initial_selected_patch_candidate_slot"] == 0
    assert result["identity_retry_slot"] == 2
    assert result["selected_patch_candidate_slot"] == 2
    assert [attempt["candidate_slot"] for attempt in result["candidate_attempts"]] == [0, 2]


def test_phase2aa_runtime_identity_sidecar_uses_runtime_evidence_not_all_candidates() -> None:
    row = phase2z_row_to_phase2aa(
        {
            "trace_id": "holdout:repo0:identity",
            "split": "holdout",
            "source_kind": "public_repo",
            "repo_id": "repo0",
            "current_visible_text": "public runtime evidence",
            "runtime_visible_evidence": {
                "changed_files": ["a.py"],
                "structural_probe_hashes": ["correct-hash"],
            },
            "repair_candidates": [
                {
                    "repair_action": "wrong",
                    "structural_probe_hash": "wrong-hash",
                    "verification_command": "pytest",
                },
                {
                    "repair_action": "correct",
                    "structural_probe_hash": "correct-hash",
                    "verification_command": "pytest",
                },
            ],
            "expected_repair_action": "correct",
            "expected_repair_result": {"test_target": "phase2z_repair_tests/test_case.py"},
            "artifact_paths": {
                "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
            },
        }
    )

    state = _state_for_public_policy(
        row=row,
        pre_test={"duration_seconds": 0.1, "exit_code": 1, "stdout": "", "stderr": ""},
        test_rel="phase2z_repair_tests/test_case.py",
    )
    identity = runtime_command_identity_signal(state)

    assert "command_identity_tokens=correct-hash" in state.terminal.stdout_delta
    assert "command_identity_tokens=wrong-hash" not in state.terminal.stdout_delta
    assert identity["command_identity_slot:1"] > identity["command_identity_slot:0"]
    assert identity["command_identity_margin"] > 0.0


def test_phase2aa_execution_audit_rejects_low_candidate_selection_accuracy(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "success": index < 8,
            "policy_loaded": True,
            "source_kind": "public_repo",
            "claim_boundary": CLAIM_BOUNDARY,
            "claim_bearing_candidate_selection_evidence": True,
            "claim_bearing_freeform_patch_evidence": False,
            "freeform_patch_generation": False,
            "sealed_feedback_used": False,
            "patch_candidate_selected_correctly": index < 8,
            "full_test_pass_rate": 1.0 if index < 8 else 0.0,
            "rollback_failure_restored": index < 8,
            "policy_open_repair_outputs": {"patch_proposal": 1},
        }
        for index in range(10)
    ]
    results = _write_jsonl(tmp_path / "results.jsonl", rows)

    report = audit_phase2aa_bounded_patch_candidate_execution(
        execution_results_jsonl=results,
        min_rows=10,
        min_success_rate=0.7,
        min_selection_accuracy=0.85,
    )

    assert report["passed"] is False
    assert report["checks"]["selection_accuracy_minimum_met"] is False


def test_phase2aa_baseline_report_blocks_learned_head_necessity_when_identity_heuristic_matches_full(
    tmp_path: Path,
) -> None:
    rows = [
        phase2z_row_to_phase2aa(
            {
                **_row(index, "holdout"),
                "runtime_visible_evidence": {
                    "changed_files": ["a.py"],
                    "structural_probe_hashes": [f"hash{index}{index % 2}"],
                },
                "expected_repair_result": {"test_target": "phase2z_repair_tests/test_case.py"},
                "artifact_paths": {
                    "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                    "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
                },
            }
        )
        for index in range(4)
    ]
    rows_jsonl = _write_jsonl(tmp_path / "rows.jsonl", rows)
    full_summary = _write_json(
        tmp_path / "full.json",
        {"rows": 4, "success_rate": 1.0, "patch_candidate_selection_accuracy": 1.0},
    )
    no_nsi_summary = _write_json(
        tmp_path / "no_nsi.json",
        {"rows": 4, "success_rate": 0.5, "patch_candidate_selection_accuracy": 0.5},
    )

    report = build_phase2aa_candidate_selection_baseline_report(
        rows_jsonl=rows_jsonl,
        output_json=tmp_path / "report.json",
        full_summary_json=full_summary,
        no_nsi_summary_json=no_nsi_summary,
    )

    assert report["baseline_metrics"]["runtime_identity_heuristic"]["accuracy"] == 1.0
    assert report["checks"]["source_overlap_below_full"] is True
    assert report["checks"]["source_overlap_identity_text_ablated_below_full"] is True
    assert report["checks"]["no_nsi_below_full"] is True
    assert report["checks"]["identity_heuristic_below_full"] is False
    assert report["interpretation"]["bounded_candidate_selection_supported"] is True
    assert report["interpretation"]["learned_head_necessity_supported"] is False
