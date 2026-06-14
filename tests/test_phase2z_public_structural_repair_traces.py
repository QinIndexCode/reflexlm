import json
import subprocess
from pathlib import Path

from reflexlm.cli.audit_phase2z_public_nonliteral_trace_gap import (
    audit_phase2z_public_nonliteral_trace_gap,
)
from reflexlm.cli.collect_phase2z_public_structural_repair_traces import (
    collect_phase2z_public_structural_repair_traces,
    _discover_structural_targets,
    _ignore_tree,
)
from reflexlm.cli.audit_phase2au_policy_required_runtime_tasks import (
    _test_has_parser_oracle,
)
from reflexlm.cli.build_phase2s_head_dataset import phase2s_repair_trace_to_head_row


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "LICENSE").write_text("MIT fixture\n", encoding="utf-8")
    (repo / "a.py").write_text(
        "import os\n\n"
        "VALUE = os.name\n\n"
        "def normalize(value):\n"
        "    return value.strip().lower()\n",
        encoding="utf-8",
    )
    (repo / "b.py").write_text(
        "from pathlib import Path\n\n"
        "def suffix(value):\n"
        "    return Path(value).suffix\n",
        encoding="utf-8",
    )
    (repo / "c.py").write_text(
        "import json\n\n"
        "def encode(value):\n"
        "    return json.dumps(value).strip()\n",
        encoding="utf-8",
    )
    (repo / "d.py").write_text(
        "def shout(value):\n"
        "    return value.strip().upper()\n",
        encoding="utf-8",
    )
    (repo / "e.py").write_text(
        "GREETING = 'hello phase2z'\n\n"
        "def greeting():\n"
        "    return GREETING\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "phase2z@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Phase2Z Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    return repo


def test_phase2z_collects_public_structural_multifile_rows(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    specs = _write_json(
        tmp_path / "specs.json",
        [
            {
                "repo_id": "repo",
                "local_path": str(repo),
                "repo_url": "https://example.invalid/repo.git",
                "split": "holdout",
                "license": "MIT",
            }
        ],
    )

    manifest = collect_phase2z_public_structural_repair_traces(
        repo_specs_json=specs,
        output_root=tmp_path / "out",
        clone_root=tmp_path / "clones",
        rows_per_repo=2,
        timeout_seconds=20,
        no_clone=True,
    )

    assert manifest["splits"]["holdout"]["rows"] == 2
    assert manifest["repo_reports"][0]["eligible_repair_mode_counts"]
    assert manifest["repo_reports"][0]["emitted_repair_mode_counts"]
    report = audit_phase2z_public_nonliteral_trace_gap(
        dataset_root=tmp_path / "out",
        split_jsonl=[tmp_path / "out" / "holdout.raw.jsonl"],
        min_rows=2,
        min_structural_nonliteral_rows=2,
        min_multifile_rows=1,
    )
    assert report["passed"] is True

    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "holdout.raw.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    generated_test = tmp_path / "out" / rows[0]["artifact_paths"]["generated_test"]
    assert generated_test.exists()
    generated_test_source = generated_test.read_text(encoding="utf-8")
    assert "REPO_ROOT = Path(__file__).resolve().parents[1]" in generated_test_source
    assert str(repo) not in generated_test_source
    head_row = phase2s_repair_trace_to_head_row(rows[0])
    slot = int(head_row["command_slot"])
    assert head_row["nsi_reference"][f"command_identity_slot:{slot}"] > 0.0
    assert head_row["nsi_reference"]["command_identity_confidence"] > 0.0
    assert head_row["patch_proposal_label"] == 1
    assert head_row["bounded_edit_scope_label"] == 1
    assert head_row["rollback_safety_label"] == 1


def test_phase2z_collects_from_pinned_local_commit(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    pinned = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    (repo / "late.py").write_text("def late():\n    return 'late'\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "late"], cwd=repo, check=True, capture_output=True)
    specs = _write_json(
        tmp_path / "specs.json",
        [
            {
                "repo_id": "repo",
                "repo_url": str(repo.resolve().as_uri()),
                "commit_hash": pinned,
                "split": "holdout",
                "license": "MIT",
            }
        ],
    )

    manifest = collect_phase2z_public_structural_repair_traces(
        repo_specs_json=specs,
        output_root=tmp_path / "out_pinned",
        clone_root=tmp_path / "clones_pinned",
        rows_per_repo=1,
        timeout_seconds=20,
    )

    clone_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path / "clones_pinned" / "repo",
        text=True,
    ).strip()
    assert clone_head == pinned
    assert manifest["splits"]["holdout"]["rows"] == 1


def test_phase2z_discovers_call_attribute_targets(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)

    targets = _discover_structural_targets(repo)

    assert any(target.repair_mode == "call_attribute_restoration" for target in targets)


def test_phase2z_stratified_selection_emits_attribute_tests(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    specs = _write_json(
        tmp_path / "specs.json",
        [
            {
                "repo_id": "repo",
                "local_path": str(repo),
                "repo_url": "https://example.invalid/repo.git",
                "split": "holdout",
                "license": "MIT",
            }
        ],
    )

    manifest = collect_phase2z_public_structural_repair_traces(
        repo_specs_json=specs,
        output_root=tmp_path / "out_stratified",
        clone_root=tmp_path / "clones_stratified",
        rows_per_repo=2,
        timeout_seconds=20,
        no_clone=True,
        target_selection_policy="stratified_repair_mode",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "out_stratified" / "holdout.raw.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    generated_tests = [
        (tmp_path / "out_stratified" / row["artifact_paths"]["generated_test"]).read_text(
            encoding="utf-8"
        )
        for row in rows
    ]
    assert manifest["target_selection_policy"] == "stratified_repair_mode"
    assert any("node.attr ==" in source for source in generated_tests)


def test_phase2z_behavioral_selection_emits_non_parser_oracle_tests(
    tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    specs = _write_json(
        tmp_path / "specs.json",
        [
            {
                "repo_id": "repo",
                "local_path": str(repo),
                "repo_url": "https://example.invalid/repo.git",
                "split": "holdout",
                "license": "MIT",
            }
        ],
    )

    manifest = collect_phase2z_public_structural_repair_traces(
        repo_specs_json=specs,
        output_root=tmp_path / "out_behavioral",
        clone_root=tmp_path / "clones_behavioral",
        rows_per_repo=1,
        timeout_seconds=20,
        no_clone=True,
        target_selection_policy="behavioral_string_method",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "out_behavioral" / "holdout.raw.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    generated_test = tmp_path / "out_behavioral" / rows[0]["artifact_paths"]["generated_test"]
    command_log = tmp_path / "out_behavioral" / rows[0]["artifact_paths"]["command_log"]
    log = json.loads(command_log.read_text(encoding="utf-8"))
    exit_codes = [entry["exit_code"] for entry in log["commands"]]

    assert manifest["target_selection_policy"] == "behavioral_string_method"
    assert manifest["splits"]["holdout"]["rows"] == 1
    assert _test_has_parser_oracle(generated_test) is False
    assert "read_text(encoding='utf-8')" not in generated_test.read_text(encoding="utf-8")
    assert exit_codes == [1, 0, 1]


def test_phase2z_behavioral_selection_interleaves_import_and_string_method_targets(
    tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    specs = _write_json(
        tmp_path / "specs.json",
        [
            {
                "repo_id": "repo",
                "local_path": str(repo),
                "repo_url": "https://example.invalid/repo.git",
                "split": "holdout",
                "license": "MIT",
            }
        ],
    )

    collect_phase2z_public_structural_repair_traces(
        repo_specs_json=specs,
        output_root=tmp_path / "out_behavioral",
        clone_root=tmp_path / "clones_behavioral",
        rows_per_repo=4,
        timeout_seconds=20,
        no_clone=True,
        target_selection_policy="behavioral_string_method",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "out_behavioral" / "holdout.raw.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    modes = {
        mode
        for row in rows
        for mode in row["runtime_visible_evidence"]["repair_modes"]
    }
    generated_tests = [
        (tmp_path / "out_behavioral" / row["artifact_paths"]["generated_test"]).read_text(
            encoding="utf-8"
        )
        for row in rows
    ]

    assert {"behavioral_import_restoration", "behavioral_string_method_restoration"}.issubset(
        modes
    )
    assert all("read_text(encoding='utf-8')" not in source for source in generated_tests)


def test_phase2z_behavioral_string_method_only_selection_excludes_import_targets(
    tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    specs = _write_json(
        tmp_path / "specs.json",
        [
            {
                "repo_id": "repo",
                "local_path": str(repo),
                "repo_url": "https://example.invalid/repo.git",
                "split": "holdout",
                "license": "MIT",
            }
        ],
    )

    collect_phase2z_public_structural_repair_traces(
        repo_specs_json=specs,
        output_root=tmp_path / "out_behavioral_string_only",
        clone_root=tmp_path / "clones_behavioral_string_only",
        rows_per_repo=1,
        timeout_seconds=20,
        no_clone=True,
        target_selection_policy="behavioral_string_method_only",
    )

    rows = [
        json.loads(line)
        for line in (
            tmp_path / "out_behavioral_string_only" / "holdout.raw.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert rows
    assert {
        mode
        for row in rows
        for mode in row["runtime_visible_evidence"]["repair_modes"]
    } == {"behavioral_string_method_restoration"}


def test_phase2z_behavioral_attribute_import_selection_emits_non_parser_oracle_call_rows(
    tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    specs = _write_json(
        tmp_path / "specs.json",
        [
            {
                "repo_id": "repo",
                "local_path": str(repo),
                "repo_url": "https://example.invalid/repo.git",
                "split": "holdout",
                "license": "MIT",
            }
        ],
    )

    collect_phase2z_public_structural_repair_traces(
        repo_specs_json=specs,
        output_root=tmp_path / "out_behavioral_attr",
        clone_root=tmp_path / "clones_behavioral_attr",
        rows_per_repo=4,
        timeout_seconds=20,
        no_clone=True,
        target_selection_policy="behavioral_attribute_import",
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "out_behavioral_attr" / "holdout.raw.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    generated_tests = [
        (tmp_path / "out_behavioral_attr" / row["artifact_paths"]["generated_test"]).read_text(
            encoding="utf-8"
        )
        for row in rows
    ]
    modes = {
        mode
        for row in rows
        for mode in row["runtime_visible_evidence"]["repair_modes"]
    }

    assert "behavioral_string_method_restoration" in modes
    assert all(
        len(set(row["runtime_visible_evidence"]["repair_modes"])) == 1 for row in rows
    )
    assert all("read_text(encoding='utf-8')" not in source for source in generated_tests)
    assert all("node.attr ==" not in source for source in generated_tests)


def test_phase2z_behavioral_diverse_descriptor_selection_emits_three_repair_modes(
    tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    specs = _write_json(
        tmp_path / "specs.json",
        [
            {
                "repo_id": "repo",
                "local_path": str(repo),
                "repo_url": "https://example.invalid/repo.git",
                "split": "holdout",
                "license": "MIT",
            }
        ],
    )

    collect_phase2z_public_structural_repair_traces(
        repo_specs_json=specs,
        output_root=tmp_path / "out_behavioral_diverse",
        clone_root=tmp_path / "clones_behavioral_diverse",
        rows_per_repo=6,
        timeout_seconds=20,
        no_clone=True,
        target_selection_policy="behavioral_diverse_descriptor",
    )

    rows = [
        json.loads(line)
        for line in (
            tmp_path / "out_behavioral_diverse" / "holdout.raw.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    modes = {
        mode
        for row in rows
        for mode in row["runtime_visible_evidence"]["repair_modes"]
    }

    assert "behavioral_string_method_restoration" in modes
    assert "behavioral_import_restoration" in modes
    assert "call_attribute_restoration" in modes
    assert modes.intersection(
        {"literal_restoration", "module_constant_literal_restoration"}
    )


def test_phase2z_copy_ignore_skips_cookiecutter_template_names() -> None:
    ignored = _ignore_tree(
        "unused",
        [
            "{{cookiecutter.repo_name}}",
            "normal_package",
            "__pycache__",
            "module.pyc",
        ],
    )

    assert "{{cookiecutter.repo_name}}" in ignored
    assert "__pycache__" in ignored
    assert "module.pyc" in ignored
    assert "normal_package" not in ignored
