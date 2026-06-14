import json
import subprocess
from pathlib import Path

from reflexlm.cli.audit_phase2av_public_repo_capacity import (
    audit_phase2av_public_repo_capacity,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text(
        "import os\n\n"
        "VALUE = os.name\n\n"
        "def normalize(value):\n"
        "    return value.strip().lower()\n\n"
        "def literal():\n"
        "    return 'ready'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "phase2av@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Phase2AV Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    return repo


def test_phase2av_public_repo_capacity_reports_non_import_targets(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    specs = _write_json(
        tmp_path / "specs.json",
        [
            {
                "repo_id": "repo",
                "local_path": str(repo),
                "repo_url": "https://example.invalid/repo.git",
                "split": "train",
            }
        ],
    )

    report = audit_phase2av_public_repo_capacity(
        repo_specs_json=specs,
        clone_root=tmp_path / "clones",
        no_clone=True,
        min_non_import_targets_per_repo=1,
        min_total_non_import_targets=1,
    )

    assert report["passed"] is True
    assert report["ready_for_phase2av_candidate_collection"] is True
    assert report["non_import_target_total"] >= 1
    assert report["recommended_repos"]
    assert "phase2av_full_training_ready" in report["unsupported_claims"]


def test_phase2av_public_repo_capacity_rejects_import_only_pool(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text("import os\nVALUE = os.name\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "phase2av@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Phase2AV Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=repo, check=True, capture_output=True)
    specs = _write_json(
        tmp_path / "specs.json",
        [{"repo_id": "repo", "local_path": str(repo), "split": "train"}],
    )

    report = audit_phase2av_public_repo_capacity(
        repo_specs_json=specs,
        clone_root=tmp_path / "clones",
        no_clone=True,
        min_non_import_targets_per_repo=1,
        min_total_non_import_targets=1,
    )

    assert report["passed"] is False
    assert "public_repo_capacity_below_non_import_descriptor_threshold" in report[
        "blocking_reasons"
    ]
