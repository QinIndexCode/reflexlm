import json
from pathlib import Path

from reflexlm.cli.audit_phase2av_descriptor_latent_oracle import (
    audit_phase2av_descriptor_latent_oracle,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(family: str, operation: int, template: int) -> dict:
    return {
        "nsi_reference": {"descriptor_failure_family": family},
        "patch_operation_label": operation,
        "patch_template_slot": template,
    }


def test_phase2av_descriptor_latent_oracle_accepts_separable_runtime_family(
    tmp_path: Path,
) -> None:
    train = [
        _row("attribute_missing_runtime", 1, 0),
        _row("attribute_missing_runtime", 1, 0),
        _row("missing_import_or_symbol_runtime", 2, 1),
        _row("missing_import_or_symbol_runtime", 2, 1),
        _row("assertion_behavior_mismatch_runtime", 3, 3),
        _row("assertion_behavior_mismatch_runtime", 3, 3),
    ]
    val = [
        _row("attribute_missing_runtime", 1, 0),
        _row("missing_import_or_symbol_runtime", 2, 1),
        _row("assertion_behavior_mismatch_runtime", 3, 3),
    ]

    report = audit_phase2av_descriptor_latent_oracle(
        train_jsonl=_write_jsonl(tmp_path / "train.jsonl", train),
        eval_jsonl=_write_jsonl(tmp_path / "val.jsonl", val),
        min_oracle_accuracy=0.85,
    )

    assert report["passed"] is True
    assert report["metrics"]["patch_operation_oracle"]["accuracy"] == 1.0
    assert report["metrics"]["patch_template_oracle"]["accuracy"] == 1.0
    assert "do_not_package_from_oracle" in report["blocked_actions"]
    assert "sealed_cross_model_transfer" in report["unsupported_claims"]


def test_phase2av_descriptor_latent_oracle_rejects_unseen_eval_family(
    tmp_path: Path,
) -> None:
    train = [
        _row("attribute_missing_runtime", 1, 0),
        _row("missing_import_or_symbol_runtime", 2, 1),
    ]
    val = [
        _row("syntax_load_failure_runtime", 0, 2),
        _row("attribute_missing_runtime", 1, 0),
    ]

    report = audit_phase2av_descriptor_latent_oracle(
        train_jsonl=_write_jsonl(tmp_path / "train.jsonl", train),
        eval_jsonl=_write_jsonl(tmp_path / "val.jsonl", val),
        min_oracle_accuracy=0.85,
    )

    assert report["passed"] is False
    assert report["metrics"]["patch_operation_oracle"]["missing_families"] == {
        "syntax_load_failure_runtime": 1
    }
