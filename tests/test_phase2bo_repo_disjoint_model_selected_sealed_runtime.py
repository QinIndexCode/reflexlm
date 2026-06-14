import subprocess
from pathlib import Path

import pytest

from reflexlm.cli.run_phase2bo_repo_disjoint_model_selected_sealed_runtime import (
    _git_repository_provenance,
    _repo_identity_checks,
)


def _init_repo(path: Path, origin: str) -> None:
    path.mkdir()
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "phase2bo@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Phase2BO Test"],
        check=True,
    )
    (path / "README.md").write_text(origin, encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "fixture"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "remote", "add", "origin", origin],
        check=True,
    )


def test_phase2bo_records_independent_git_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source"
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    _init_repo(source, "https://example.invalid/source.git")
    _init_repo(repo_a, "https://example.invalid/a.git")
    _init_repo(repo_b, "https://example.invalid/b.git")

    source_provenance = _git_repository_provenance(source)
    provenances = [_git_repository_provenance(repo_a), _git_repository_provenance(repo_b)]
    checks = _repo_identity_checks(
        source=source_provenance,
        repositories=provenances,
        minimum_repository_count=2,
    )

    assert all(checks.values())
    assert provenances[0]["origin"] != provenances[1]["origin"]


def test_phase2bo_rejects_workspace_below_git_root(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    _init_repo(repository, "https://example.invalid/repo.git")
    nested = repository / "nested"
    nested.mkdir()

    with pytest.raises(ValueError, match="must be a git root"):
        _git_repository_provenance(nested)
