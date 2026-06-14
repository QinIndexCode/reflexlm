import json
from pathlib import Path

from reflexlm.cli.build_phase2cg_unified_verification_package import (
    build_phase2cg_unified_verification_package,
)


class _Summary:
    training_top1_accuracy = 1.0


class _BaseMatcher:
    training_summary = _Summary()

    def save(self, path: Path) -> Path:
        path.write_bytes(b"checkpoint")
        return path


class _Matcher:
    matcher = _BaseMatcher()

    def metadata(self) -> dict:
        return {"matcher_family": "test"}


def test_phase2cg_builder_packages_internal_verification_cortex(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = tmp_path / "base"
    package.mkdir()
    (package / "native_nervous_package.json").write_text(
        json.dumps(
            {
                "package_family": "phase2d_native_nervous_package",
                "policy_label": "base",
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {"repo_origin": f"repo-{repo}", "pre": {}, "post": {}}
        for repo in range(6)
        for _ in range(2)
    ]
    monkeypatch.setattr(
        "reflexlm.cli.build_phase2cg_unified_verification_package._execution_rows",
        lambda *_args: rows,
    )
    monkeypatch.setattr(
        "reflexlm.cli.build_phase2cg_unified_verification_package._train_verification_matcher",
        lambda **_kwargs: _Matcher(),
    )

    report = build_phase2cg_unified_verification_package(
        base_package_path=package,
        historical_execution_jsonl="execution.jsonl",
        tasks_jsonl="tasks.jsonl",
        cortex_model_path="cortex",
        cortex_device="cpu",
        cortex_dtype="float32",
        output_package_dir=tmp_path / "output",
        output_report_json=tmp_path / "report.json",
    )

    manifest = json.loads(
        (tmp_path / "output" / "native_nervous_package.json").read_text()
    )
    assert report["passed"] is True
    assert manifest["verification_control_source"] == (
        "package_internal_verification_cortex"
    )
    assert Path(manifest["verification_cortex_path"]).is_file()
