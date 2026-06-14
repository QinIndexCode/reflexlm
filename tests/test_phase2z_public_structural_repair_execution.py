import json
import subprocess
from pathlib import Path

from reflexlm.cli.run_phase2z_public_structural_repair_execution import (
    _insert_required_text,
    _parse_missing_import_requirements_from_stdout,
    _parse_stdout_literal_replacements,
    run_phase2z_public_structural_repair_execution,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _make_public_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "clones" / "repo0"
    repo.mkdir(parents=True)
    (repo / "a.py").write_text("import os\n\nVALUE = os.name\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "phase2z@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Phase2Z Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    return repo


def test_phase2z_import_insertion_preserves_future_import_order() -> None:
    source = (
        "# Copyright header\n"
        "# More header\n"
        "from __future__ import annotations\n"
        "\n"
        "VALUE = Final[int]\n"
    )

    patched = _insert_required_text(source, "from typing import Final")

    assert patched.splitlines()[:4] == [
        "# Copyright header",
        "# More header",
        "from __future__ import annotations",
        "",
    ]
    assert "from typing import Final\nVALUE = Final[int]" in patched


def test_phase2z_import_insertion_ignores_string_literal_mentions() -> None:
    source = (
        "#! /bin/env python\n"
        "import dataclasses\n"
        "\n"
        "CONFIG = '''\n"
        "add_imports=import os,import json\n"
        "'''\n"
        "OUTPUT_FILE = os.path.abspath('out.txt')\n"
    )

    patched = _insert_required_text(source, "import os")

    assert patched.splitlines()[:2] == ["#! /bin/env python", "import os"]
    assert patched.count("import os") == 2


def test_phase2z_missing_import_resolver_handles_runtime_aliases() -> None:
    stdout = (
        "pkg/module.py:10: NameError\n"
        "E   NameError: name '_t' is not defined\n"
        "pkg/plugin.py:12: NameError\n"
        "E   NameError: name 'pluggy' is not defined\n"
        "pkg/types.py:13: NameError\n"
        "E   NameError: name 'TypedDict' is not defined\n"
        "pkg/types.py:14: NameError\n"
        "E   NameError: name 'cast' is not defined\n"
    )
    evidence = {"changed_files": ["pkg/module.py", "pkg/plugin.py", "pkg/types.py"]}

    requirements = _parse_missing_import_requirements_from_stdout(stdout, evidence)

    assert ("pkg/module.py", "import typing as _t") in requirements
    assert ("pkg/plugin.py", "import pluggy") in requirements
    assert ("pkg/types.py", "from typing import TypedDict") in requirements
    assert ("pkg/types.py", "from typing import cast") in requirements


def test_phase2z_missing_import_resolver_uses_traceback_file_locations() -> None:
    stderr = (
        "  File \"C:\\tmp\\sandbox\\src\\pluggy\\_result.py\", line 15, in <module>\n"
        "    _ExcInfo: TypeAlias = tuple[type[BaseException], BaseException, TracebackType | None]\n"
        "                                                                    ^^^^^^^^^^^^^\n"
        "NameError: name 'TracebackType' is not defined\n"
    )
    evidence = {
        "changed_files": ["src/pluggy/_result.py", "src/pluggy/_hooks.py"]
    }

    requirements = _parse_missing_import_requirements_from_stdout(stderr, evidence)

    assert requirements == [
        ("src/pluggy/_result.py", "from types import TracebackType")
    ]


def test_phase2z_literal_replacements_restore_unknown_phase2z_encoding() -> None:
    stdout = (
        "E       LookupError: unknown encoding: latin1_phase2z_mutated\n"
        "src\\werkzeug\\_internal.py:34: LookupError\n"
    )

    assert _parse_stdout_literal_replacements(stdout) == [
        ("latin1_phase2z_mutated", "latin1")
    ]


def test_phase2z_public_structural_runner_replays_recorded_patch_with_boundary(
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
    (artifact_dir / "generated_test.py").write_text(
        "from pathlib import Path\n\n"
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n\n"
        "def test_import_restored():\n"
        "    assert 'import os' in (REPO_ROOT / 'a.py').read_text(encoding='utf-8')\n",
        encoding="utf-8",
    )
    rows = _write_jsonl(
        tmp_path / "rows.jsonl",
        [
            {
                "trace_id": "holdout:repo0:0",
                "split": "holdout",
                "source_kind": "public_repo",
                "repo_id": "repo0",
                "repo_url_or_origin": "https://example.invalid/repo0.git",
                "commit_hash": "a" * 40,
                "runtime_visible_evidence": {"changed_files": ["a.py"]},
                "repair_candidates": [{"verification_command": "python -m pytest -q phase2z_repair_tests/test_case.py"}],
                "expected_repair_result": {"test_target": "phase2z_repair_tests/test_case.py"},
                "artifact_paths": {
                    "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                    "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
                },
            }
        ],
    )
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

    report = run_phase2z_public_structural_repair_execution(
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
    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert result["source_kind"] == "public_repo"
    assert result["claim_bearing_execution_evidence"] is False
    assert result["claim_boundary"] == "public_structural_recorded_patch_runtime_control_only_not_model_patch_generation"
    assert result["rollback_failure_restored"] is True


def test_phase2z_public_structural_runner_generates_runtime_symbolic_patch(
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
    (artifact_dir / "generated_test.py").write_text(
        "from pathlib import Path\n\n"
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n\n"
        "def test_import_restored():\n"
        "    text = (REPO_ROOT / 'a.py').read_text(encoding='utf-8')\n"
        "    assert 'import os' in text\n",
        encoding="utf-8",
    )
    rows = _write_jsonl(
        tmp_path / "rows.jsonl",
        [
            {
                "trace_id": "holdout:repo0:0",
                "split": "holdout",
                "source_kind": "public_repo",
                "repo_id": "repo0",
                "repo_url_or_origin": "https://example.invalid/repo0.git",
                "commit_hash": "a" * 40,
                "runtime_visible_evidence": {"changed_files": ["a.py"]},
                "repair_candidates": [{"verification_command": "python -m pytest -q phase2z_repair_tests/test_case.py"}],
                "expected_repair_result": {"test_target": "phase2z_repair_tests/test_case.py"},
                "artifact_paths": {
                    "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                    "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
                },
            }
        ],
    )
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

    report = run_phase2z_public_structural_repair_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
        patch_mode="runtime_symbolic_membership",
    )

    assert report["successes"] == 1
    assert report["patch_mode"] == "runtime_symbolic_membership"
    assert (
        report["claim_boundary"]
        == "bounded_runtime_symbolic_patch_proposal_only_not_open_ended_repair"
    )
    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert result["patch_source"] == "package_runtime_symbolic_text_membership_patch_proposal"
    assert result["patch_generator"] == "bounded_symbolic_text_membership_patch_v1"
    assert (
        result["claim_boundary"]
        == "bounded_runtime_symbolic_patch_proposal_only_not_open_ended_repair"
    )
    assert result["claim_bearing_execution_evidence"] is True
    assert result["recorded_patch_artifact_used"] is False
    assert result["recorded_patch_artifact_used_for_fault_injection"] is True


def test_phase2z_public_structural_runner_reports_authorized_patch_generation_failure(
    tmp_path: Path,
) -> None:
    _make_public_repo(tmp_path)
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
        "def test_runtime_failure_without_symbolic_requirements():\n"
        "    assert False\n",
        encoding="utf-8",
    )
    rows = _write_jsonl(
        tmp_path / "rows.jsonl",
        [
            {
                "trace_id": "holdout:repo0:0",
                "split": "holdout",
                "source_kind": "public_repo",
                "repo_id": "repo0",
                "repo_url_or_origin": "https://example.invalid/repo0.git",
                "commit_hash": "a" * 40,
                "runtime_visible_evidence": {"changed_files": ["a.py"]},
                "repair_candidates": [
                    {
                        "verification_command": (
                            "python -m pytest -q phase2z_repair_tests/test_case.py"
                        )
                    }
                ],
                "expected_repair_result": {
                    "test_target": "phase2z_repair_tests/test_case.py"
                },
                "artifact_paths": {
                    "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                    "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
                },
            }
        ],
    )
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

    report = run_phase2z_public_structural_repair_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
        patch_mode="runtime_symbolic_structural",
    )

    assert report["successes"] == 0
    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert result["patch_authorized"] is True
    assert result["symbolic_patch_failure"] == "missing_symbolic_structural_requirements"
    assert result["patch_source"] == "package_runtime_symbolic_patch_unavailable"
    patch_apply = json.loads(
        Path(result["artifact_paths"]["patch_apply_log"]).read_text(encoding="utf-8")
    )
    assert patch_apply["stderr"] == "patch_generation_failed"
    assert (
        patch_apply["symbolic_patch_failure"]
        == "missing_symbolic_structural_requirements"
    )


def test_phase2z_public_structural_runner_generates_runtime_structural_attribute_patch(
    tmp_path: Path,
) -> None:
    repo = _make_public_repo(tmp_path)
    (repo / "module.py").write_text(
        "def normalize(value):\n    return value.lower()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add module"], cwd=repo, check=True, capture_output=True)
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    test_rel = "phase2z_repair_tests/test_phase2z_structural_case_00000.py"
    test_source = "\n".join(
        [
            "import ast",
            "from pathlib import Path",
            "",
            "REPO_ROOT = Path(__file__).resolve().parents[1]",
            "",
            "def test_attr_restored():",
            "    tree = ast.parse((REPO_ROOT / 'module.py').read_text(encoding='utf-8'))",
            "    assert any(isinstance(node, ast.Attribute) and node.attr == 'lower' for node in ast.walk(tree))",
            "",
        ]
    )
    (artifact_dir / "generated_test.py").write_text(test_source, encoding="utf-8")
    patch = "\n".join(
        [
            "--- a/module.py",
            "+++ b/module.py",
            "@@ -1,2 +1,2 @@",
            " def normalize(value):",
            "-    return value.phase2z_missing_lower()",
            "+    return value.lower()",
            "",
        ]
    )
    (artifact_dir / "patch.diff").write_text(patch, encoding="utf-8")
    rows = _write_jsonl(
        dataset / "holdout.raw.jsonl",
        [
            {
                "trace_id": "holdout:repo0:attr",
                "repo_id": "repo0",
                "source_kind": "public_repo",
                "repo_url_or_origin": "https://example.invalid/repo0.git",
                "commit_hash": "abc123",
                "runtime_visible_evidence": {
                    "changed_files": ["module.py"],
                    "structural_probe_hashes": ["probe-attr"],
                },
                "repair_candidates": [{"repair_action": "repair_attr"}],
                "expected_repair_result": {"test_target": test_rel},
                "artifact_paths": {
                    "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                    "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
                },
            }
        ],
    )
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

    report = run_phase2z_public_structural_repair_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
        patch_mode="runtime_symbolic_structural",
    )

    assert report["successes"] == 1
    assert (
        report["claim_boundary"]
        == "bounded_runtime_symbolic_structural_patch_proposal_only_not_open_ended_repair"
    )
    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert result["patch_source"] == "package_runtime_symbolic_structural_patch_proposal"
    assert result["patch_generator"] == "bounded_symbolic_structural_patch_v1"
    assert result["symbolic_patch_kinds"] == ["ast_attribute_restoration"]
    assert result["recorded_patch_artifact_used"] is False


def test_phase2z_public_structural_runner_generates_attribute_patch_from_traceback(
    tmp_path: Path,
) -> None:
    repo = _make_public_repo(tmp_path)
    (repo / "module.py").write_text(
        "def normalize(value):\n    return value.lower()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add traceback module"], cwd=repo, check=True, capture_output=True)
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    test_rel = "phase2z_repair_tests/test_phase2z_structural_case_00000.py"
    (artifact_dir / "generated_test.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import importlib.util",
                "REPO_ROOT = Path(__file__).resolve().parents[1]",
                "def test_attr_restored_from_runtime_traceback():",
                "    spec = importlib.util.spec_from_file_location('module', REPO_ROOT / 'module.py')",
                "    module = importlib.util.module_from_spec(spec)",
                "    assert spec.loader is not None",
                "    spec.loader.exec_module(module)",
                "    assert module.normalize('ABC') == 'abc'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (artifact_dir / "patch.diff").write_text(
        "--- a/module.py\n"
        "+++ b/module.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def normalize(value):\n"
        "-    return value.phase2z_missing_lower()\n"
        "+    return value.lower()\n",
        encoding="utf-8",
    )
    rows = _write_jsonl(
        dataset / "holdout.raw.jsonl",
        [
            {
                "trace_id": "holdout:repo0:traceback-attr",
                "repo_id": "repo0",
                "source_kind": "public_repo",
                "repo_url_or_origin": "https://example.invalid/repo0.git",
                "commit_hash": "abc123",
                "runtime_visible_evidence": {"changed_files": ["module.py"]},
                "repair_candidates": [{"repair_action": "repair_attr"}],
                "expected_repair_result": {"test_target": test_rel},
                "artifact_paths": {
                    "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                    "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
                },
            }
        ],
    )
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

    report = run_phase2z_public_structural_repair_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
        patch_mode="runtime_symbolic_structural",
    )

    assert report["successes"] == 1
    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert result["symbolic_patch_kinds"] == ["ast_attribute_restoration"]
    assert "phase2z_missing_lower" in Path(result["artifact_paths"]["patch"]).read_text(
        encoding="utf-8"
    )


def test_phase2z_public_structural_runner_generates_import_patch_from_name_error(
    tmp_path: Path,
) -> None:
    repo = _make_public_repo(tmp_path)
    (repo / "module.py").write_text(
        "import os\nVALUE = os.getenv('X')\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add nameerror module"], cwd=repo, check=True, capture_output=True)
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    test_rel = "phase2z_repair_tests/test_phase2z_structural_case_00000.py"
    (artifact_dir / "generated_test.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import importlib.util",
                "REPO_ROOT = Path(__file__).resolve().parents[1]",
                "def test_import_restored_from_runtime_traceback():",
                "    spec = importlib.util.spec_from_file_location('module', REPO_ROOT / 'module.py')",
                "    module = importlib.util.module_from_spec(spec)",
                "    assert spec.loader is not None",
                "    spec.loader.exec_module(module)",
                "    assert module.VALUE is None",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (artifact_dir / "patch.diff").write_text(
        "--- a/module.py\n"
        "+++ b/module.py\n"
        "@@ -1 +1,2 @@\n"
        "+import os\n"
        " VALUE = os.getenv('X')\n",
        encoding="utf-8",
    )
    rows = _write_jsonl(
        dataset / "holdout.raw.jsonl",
        [
            {
                "trace_id": "holdout:repo0:nameerror-import",
                "repo_id": "repo0",
                "source_kind": "public_repo",
                "repo_url_or_origin": "https://example.invalid/repo0.git",
                "commit_hash": "abc123",
                "runtime_visible_evidence": {"changed_files": ["module.py"]},
                "repair_candidates": [{"repair_action": "repair_import"}],
                "expected_repair_result": {"test_target": test_rel},
                "artifact_paths": {
                    "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                    "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
                },
            }
        ],
    )
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

    report = run_phase2z_public_structural_repair_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
        patch_mode="runtime_symbolic_structural",
    )

    assert report["successes"] == 1
    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert result["symbolic_patch_kinds"] == ["import_restoration"]


def test_phase2z_public_structural_runner_generates_literal_patch_from_assertion(
    tmp_path: Path,
) -> None:
    repo = _make_public_repo(tmp_path)
    (repo / "module.py").write_text("VALUE = 88\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add literal module"], cwd=repo, check=True, capture_output=True)
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    test_rel = "phase2z_repair_tests/test_phase2z_structural_case_00000.py"
    (artifact_dir / "generated_test.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import importlib.util",
                "REPO_ROOT = Path(__file__).resolve().parents[1]",
                "def _load_module(name, path):",
                "    spec = importlib.util.spec_from_file_location(name, path)",
                "    module = importlib.util.module_from_spec(spec)",
                "    assert spec.loader is not None",
                "    spec.loader.exec_module(module)",
                "    return module",
                "def test_literal_restored():",
                "    module = _load_module('module', (REPO_ROOT / 'module.py'))",
                "    observed = getattr(module, 'VALUE')",
                "    assert observed == 88",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (artifact_dir / "patch.diff").write_text(
        "--- a/module.py\n+++ b/module.py\n@@ -1 +1 @@\n-VALUE = 89\n+VALUE = 88\n",
        encoding="utf-8",
    )
    rows = _write_jsonl(
        dataset / "holdout.raw.jsonl",
        [
            {
                "trace_id": "holdout:repo0:literal",
                "repo_id": "repo0",
                "source_kind": "public_repo",
                "repo_url_or_origin": "https://example.invalid/repo0.git",
                "commit_hash": "abc123",
                "runtime_visible_evidence": {"changed_files": ["module.py"]},
                "repair_candidates": [{"repair_action": "repair_literal"}],
                "expected_repair_result": {"test_target": test_rel},
                "artifact_paths": {
                    "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                    "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
                },
            }
        ],
    )
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

    report = run_phase2z_public_structural_repair_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
        patch_mode="runtime_symbolic_structural",
    )

    assert report["successes"] == 1
    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert result["symbolic_patch_kinds"] == ["literal_restoration"]


def test_phase2z_public_structural_runner_restores_importlib_metadata_alias(
    tmp_path: Path,
) -> None:
    repo = _make_public_repo(tmp_path)
    (repo / "version.py").write_text(
        "from importlib import metadata\n\nVALUE = metadata\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add version"], cwd=repo, check=True, capture_output=True)
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    test_rel = "phase2z_repair_tests/test_phase2z_structural_case_00000.py"
    (artifact_dir / "generated_test.py").write_text(
        "\n".join(
            [
                "import importlib.util",
                "from pathlib import Path",
                "REPO_ROOT = Path(__file__).resolve().parents[1]",
                "def _load_module(name, path):",
                "    spec = importlib.util.spec_from_file_location(name, path)",
                "    module = importlib.util.module_from_spec(spec)",
                "    spec.loader.exec_module(module)",
                "    return module",
                "def test_metadata_alias_restored():",
                "    module = _load_module('fixture_version', REPO_ROOT / 'version.py')",
                "    assert module.VALUE is not None",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (artifact_dir / "patch.diff").write_text(
        "--- a/version.py\n"
        "+++ b/version.py\n"
        "@@ -1,2 +1,3 @@\n"
        "+from importlib import metadata\n"
        " \n"
        " VALUE = metadata\n",
        encoding="utf-8",
    )
    rows = _write_jsonl(
        dataset / "holdout.raw.jsonl",
        [
            {
                "trace_id": "holdout:repo0:metadata",
                "repo_id": "repo0",
                "source_kind": "public_repo",
                "repo_url_or_origin": "https://example.invalid/repo0.git",
                "commit_hash": "abc123",
                "runtime_visible_evidence": {"changed_files": ["version.py"]},
                "repair_candidates": [{"repair_action": "repair_import"}],
                "expected_repair_result": {"test_target": test_rel},
                "artifact_paths": {
                    "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                    "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
                },
            }
        ],
    )
    package = _write_json(
        tmp_path / "package" / "native_nervous_package.json",
        {
            "policy_label": "test",
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

    report = run_phase2z_public_structural_repair_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
        patch_mode="runtime_symbolic_structural",
    )

    assert report["successes"] == 1
    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    patch_text = Path(result["artifact_paths"]["patch"]).read_text(encoding="utf-8")
    assert "from importlib import metadata" in patch_text


def test_phase2z_public_structural_runner_infers_related_missing_known_imports(
    tmp_path: Path,
) -> None:
    repo = _make_public_repo(tmp_path)
    (repo / "bench.py").write_text(
        "from typing import TYPE_CHECKING\nimport os\n\nif TYPE_CHECKING:\n"
        "    from collections.abc import Callable\nVALUE = os.name\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add bench"], cwd=repo, check=True, capture_output=True)
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    test_rel = "phase2z_repair_tests/test_phase2z_structural_case_00000.py"
    (artifact_dir / "generated_test.py").write_text(
        "\n".join(
            [
                "import importlib.util",
                "from pathlib import Path",
                "REPO_ROOT = Path(__file__).resolve().parents[1]",
                "def _load_module(name, path):",
                "    spec = importlib.util.spec_from_file_location(name, path)",
                "    module = importlib.util.module_from_spec(spec)",
                "    spec.loader.exec_module(module)",
                "    return module",
                "def test_typing_and_os_imports_restored():",
                "    module = _load_module('fixture_bench', REPO_ROOT / 'bench.py')",
                "    assert isinstance(module.VALUE, str)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (artifact_dir / "patch.diff").write_text(
        "--- a/bench.py\n"
        "+++ b/bench.py\n"
        "@@ -1,3 +1,6 @@\n"
        "+from typing import TYPE_CHECKING\n"
        "+import os\n"
        "+\n"
        " if TYPE_CHECKING:\n"
        "     from collections.abc import Callable\n"
        " VALUE = os.name\n",
        encoding="utf-8",
    )
    rows = _write_jsonl(
        dataset / "holdout.raw.jsonl",
        [
            {
                "trace_id": "holdout:repo0:typing-os",
                "repo_id": "repo0",
                "source_kind": "public_repo",
                "repo_url_or_origin": "https://example.invalid/repo0.git",
                "commit_hash": "abc123",
                "runtime_visible_evidence": {"changed_files": ["bench.py"]},
                "repair_candidates": [{"repair_action": "repair_imports"}],
                "expected_repair_result": {"test_target": test_rel},
                "artifact_paths": {
                    "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                    "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
                },
            }
        ],
    )
    package = _write_json(
        tmp_path / "package" / "native_nervous_package.json",
        {
            "policy_label": "test",
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

    report = run_phase2z_public_structural_repair_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
        patch_mode="runtime_symbolic_structural",
    )

    assert report["successes"] == 1
    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    patch_text = Path(result["artifact_paths"]["patch"]).read_text(encoding="utf-8")
    assert "TYPE_CHECKING" in patch_text
    assert "import os" in patch_text


def test_phase2z_public_structural_runner_replaces_multiline_literal_assignment(
    tmp_path: Path,
) -> None:
    repo = _make_public_repo(tmp_path)
    (repo / "constants.py").write_text(
        "URL = (\n    'https://example.invalid/latest'\n)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add constants"], cwd=repo, check=True, capture_output=True)
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    test_rel = "phase2z_repair_tests/test_phase2z_structural_case_00000.py"
    (artifact_dir / "generated_test.py").write_text(
        "\n".join(
            [
                "import importlib.util",
                "from pathlib import Path",
                "REPO_ROOT = Path(__file__).resolve().parents[1]",
                "def _load_module(name, path):",
                "    spec = importlib.util.spec_from_file_location(name, path)",
                "    module = importlib.util.module_from_spec(spec)",
                "    spec.loader.exec_module(module)",
                "    return module",
                "def test_multiline_constant_restored():",
                "    module = _load_module('fixture_constants', (REPO_ROOT / 'constants.py'))",
                "    observed = getattr(module, 'URL')",
                "    assert observed == 'https://example.invalid/latest'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (artifact_dir / "patch.diff").write_text(
        "--- a/constants.py\n"
        "+++ b/constants.py\n"
        "@@ -1,3 +1,3 @@\n"
        " URL = (\n"
        "-    'https://example.invalid/latest_phase2z_mutated'\n"
        "+    'https://example.invalid/latest'\n"
        " )\n",
        encoding="utf-8",
    )
    rows = _write_jsonl(
        dataset / "holdout.raw.jsonl",
        [
            {
                "trace_id": "holdout:repo0:literal-multiline",
                "repo_id": "repo0",
                "source_kind": "public_repo",
                "repo_url_or_origin": "https://example.invalid/repo0.git",
                "commit_hash": "abc123",
                "runtime_visible_evidence": {"changed_files": ["constants.py"]},
                "repair_candidates": [{"repair_action": "repair_literal"}],
                "expected_repair_result": {"test_target": test_rel},
                "artifact_paths": {
                    "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                    "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
                },
            }
        ],
    )
    package = _write_json(
        tmp_path / "package" / "native_nervous_package.json",
        {
            "policy_label": "test",
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

    report = run_phase2z_public_structural_repair_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
        patch_mode="runtime_symbolic_structural",
    )

    assert report["successes"] == 1
    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    patch_text = Path(result["artifact_paths"]["patch"]).read_text(encoding="utf-8")
    assert "URL = 'https://example.invalid/latest'" in patch_text


def test_phase2z_public_structural_runner_generates_composite_structural_patch(
    tmp_path: Path,
) -> None:
    repo = _make_public_repo(tmp_path)
    (repo / "module.py").write_text(
        "def normalize(value):\n    return value.lower()\n",
        encoding="utf-8",
    )
    (repo / "pkg.py").write_text(
        '"""Package."""\n\nfrom __future__ import annotations\nVALUE = 1\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add composite module"], cwd=repo, check=True, capture_output=True)
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    test_rel = "phase2z_repair_tests/test_phase2z_structural_case_00000.py"
    test_source = "\n".join(
        [
            "import ast",
            "from pathlib import Path",
            "",
            "REPO_ROOT = Path(__file__).resolve().parents[1]",
            "",
            "def test_attr_restored():",
            "    text = (REPO_ROOT / 'module.py').read_text(encoding='utf-8')",
            "    assert 'phase2z_missing_lower' not in text",
            "    tree = ast.parse(text)",
            "    assert any(isinstance(node, ast.Attribute) and node.attr == 'lower' for node in ast.walk(tree))",
            "",
            "def test_import_restored():",
            "    text = (REPO_ROOT / 'pkg.py').read_text(encoding='utf-8')",
            "    assert 'from __future__ import annotations' in text",
            "",
        ]
    )
    (artifact_dir / "generated_test.py").write_text(test_source, encoding="utf-8")
    (artifact_dir / "patch.diff").write_text(
        "--- a/module.py\n"
        "+++ b/module.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def normalize(value):\n"
        "-    return value.phase2z_missing_lower()\n"
        "+    return value.lower()\n"
        "--- a/pkg.py\n"
        "+++ b/pkg.py\n"
        "@@ -1,3 +1,4 @@\n"
        " \"\"\"Package.\"\"\"\n"
        " \n"
        "+from __future__ import annotations\n"
        " VALUE = 1\n",
        encoding="utf-8",
    )
    rows = _write_jsonl(
        dataset / "holdout.raw.jsonl",
        [
            {
                "trace_id": "holdout:repo0:composite",
                "repo_id": "repo0",
                "source_kind": "public_repo",
                "repo_url_or_origin": "https://example.invalid/repo0.git",
                "commit_hash": "abc123",
                "runtime_visible_evidence": {
                    "changed_files": ["module.py", "pkg.py"],
                    "repair_modes": ["call_attribute_restoration", "import_restoration"],
                    "structural_probe_hashes": ["probe-attr", "probe-import"],
                },
                "repair_candidates": [{"repair_action": "repair_composite"}],
                "expected_repair_result": {"test_target": test_rel},
                "artifact_paths": {
                    "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                    "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
                },
            }
        ],
    )
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

    report = run_phase2z_public_structural_repair_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
        patch_mode="runtime_symbolic_structural",
    )

    assert report["successes"] == 1
    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert result["symbolic_patch_kinds"] == [
        "ast_attribute_restoration",
        "text_membership",
    ]
    assert result["patch_stats"]["changed_file_count"] == 2
    assert result["recorded_patch_artifact_used"] is False


def test_phase2z_public_structural_runner_marks_restricted_controls_non_claim(
    tmp_path: Path,
) -> None:
    repo = _make_public_repo(tmp_path)
    (repo / "module.py").write_text(
        "def normalize(value):\n    return value.lower()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add module"], cwd=repo, check=True, capture_output=True)
    dataset = tmp_path / "dataset"
    artifact_dir = dataset / "artifacts" / "holdout" / "repo0" / "row_00000"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    test_rel = "phase2z_repair_tests/test_phase2z_structural_case_00000.py"
    (artifact_dir / "generated_test.py").write_text(
        "\n".join(
            [
                "import ast",
                "from pathlib import Path",
                "REPO_ROOT = Path(__file__).resolve().parents[1]",
                "def test_attr_restored():",
                "    text = (REPO_ROOT / 'module.py').read_text(encoding='utf-8')",
                "    assert 'phase2z_missing_lower' not in text",
                "    tree = ast.parse(text)",
                "    assert any(isinstance(node, ast.Attribute) and node.attr == 'lower' for node in ast.walk(tree))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (artifact_dir / "patch.diff").write_text(
        "--- a/module.py\n"
        "+++ b/module.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def normalize(value):\n"
        "-    return value.phase2z_missing_lower()\n"
        "+    return value.lower()\n",
        encoding="utf-8",
    )
    rows = _write_jsonl(
        dataset / "holdout.raw.jsonl",
        [
            {
                "trace_id": "holdout:repo0:attr-control",
                "repo_id": "repo0",
                "source_kind": "public_repo",
                "repo_url_or_origin": "https://example.invalid/repo0.git",
                "commit_hash": "abc123",
                "runtime_visible_evidence": {
                    "changed_files": ["module.py"],
                    "structural_probe_hashes": ["probe-attr"],
                },
                "repair_candidates": [{"repair_action": "repair_attr"}],
                "expected_repair_result": {"test_target": test_rel},
                "artifact_paths": {
                    "patch_diff": "artifacts/holdout/repo0/row_00000/patch.diff",
                    "generated_test": "artifacts/holdout/repo0/row_00000/generated_test.py",
                },
            }
        ],
    )
    package = _write_json(
        tmp_path / "package" / "native_nervous_package.json",
        {
            "policy_label": "test-policy",
            "open_repair_capabilities": {
                "patch_proposal": True,
                "bounded_edit_scope": True,
                "rollback_safety": True,
            },
        },
    )

    report = run_phase2z_public_structural_repair_execution(
        source_rows_jsonl=rows,
        dataset_root=dataset,
        clone_root=tmp_path / "clones",
        package_path=package,
        output_jsonl=tmp_path / "results.jsonl",
        artifact_root=tmp_path / "artifacts",
        max_rows=1,
        timeout_seconds=20,
        load_policy=False,
        patch_mode="runtime_symbolic_attribute_control",
    )

    assert report["successes"] == 1
    result = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert result["patch_source"] == "control_runtime_symbolic_ast_attribute_patch"
    assert result["claim_bearing_execution_evidence"] is False
    assert result["control_execution_evidence"] is True
