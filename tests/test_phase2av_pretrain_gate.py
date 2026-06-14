import json
from pathlib import Path

from reflexlm.cli.audit_phase2av_pretrain_gate import build_phase2av_pretrain_gate


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
HASH_F = "f" * 64


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _manifest(tmp_path: Path, *, hash_train: str = HASH_A) -> Path:
    return _write(
        tmp_path / "descriptor_manifest.json",
        {
            "artifact_family": "phase2at_learned_patch_candidate_split",
            "schema_version": "phase2at.learned_bounded_patch_candidate.v1",
            "split_counts": {"train": 14, "val": 8, "holdout": 7},
            "split_hashes": {"train": hash_train, "val": HASH_B, "holdout": HASH_C},
            "freeform_patch_generation": False,
            "recorded_patch_artifact_as_generation_target": False,
            "symbolic_generator_as_generation_target": False,
            "sealed_feedback_used": False,
        },
    )


def _build_summary(tmp_path: Path, split: str, count: int, hash_value: str) -> Path:
    return _write(
        tmp_path / f"{split}_build.json",
        {
            "artifact_family": "phase2av_graded_descriptor_runtime_task_builder",
            "passed": True,
            "source_row_count": count,
            "converted_row_count": count,
            "operation_template_pair_count": 2,
            "task_split_sha256": hash_value,
        },
    )


def _data_health(
    tmp_path: Path,
    split: str,
    count: int,
    *,
    parser_oracle: bool = False,
    blocked: bool = False,
) -> Path:
    return _write(
        tmp_path / f"{split}_data_health.json",
        {
            "artifact_family": "phase2av_graded_descriptor_runtime_data_health",
            "passed": not parser_oracle and not blocked,
            "claim_boundary": "phase2av_graded_descriptor_runtime_ready_for_pretrain_gate",
            "checks": {
                "all_rows_public_repo": True,
                "no_sealed_feedback": True,
                "no_candidate_or_gold_markers": True,
                "contract_blocks_forbidden_targets": True,
                "operation_template_diversity_met": True,
                "generated_tests_not_parser_oracle_solvable": not parser_oracle,
            },
            "metrics": {
                "row_count": count,
                "repo_origin_count": 2,
                "operation_template_pair_count": 2,
                "parser_oracle_rows": ["row0"] if parser_oracle else [],
            },
            "blocked_actions": ["do_not_train_phase2av"] if blocked else [],
        },
    )


def _report(tmp_path: Path, *, parser_oracle_val: bool = False, bad_hash: bool = False):
    manifest = _manifest(tmp_path, hash_train="not-a-hash" if bad_hash else HASH_A)
    builds = {
        "train": _build_summary(tmp_path, "train", 14, HASH_D),
        "val": _build_summary(tmp_path, "val", 8, HASH_E),
        "holdout": _build_summary(tmp_path, "holdout", 7, HASH_F),
    }
    health = {
        "train": _data_health(tmp_path, "train", 14),
        "val": _data_health(tmp_path, "val", 8, parser_oracle=parser_oracle_val),
        "holdout": _data_health(tmp_path, "holdout", 7),
    }
    return build_phase2av_pretrain_gate(
        descriptor_manifest_json=manifest,
        runtime_build_jsons=builds,
        data_health_jsons=health,
        min_train_rows=14,
        min_val_rows=8,
        min_holdout_rows=7,
    )


def test_phase2av_pretrain_gate_accepts_hash_bound_non_parser_oracle_splits(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path)

    assert report["passed"] is True
    assert report["ready_for_phase2av_smoke_training"] is True
    assert report["ready_for_phase2av_full_training"] is False
    assert (
        "phase2av_ready_for_small_scale_nonsealed_descriptor_runtime_smoke_training"
        in report["supported_claims"]
    )
    assert "learned_descriptor_runtime_delta" in report["unsupported_claims"]


def test_phase2av_pretrain_gate_rejects_parser_oracle_split(tmp_path: Path) -> None:
    report = _report(tmp_path, parser_oracle_val=True)

    assert report["passed"] is False
    assert report["checks"]["val_data_health_non_parser_oracle"] is False
    assert "do_not_train_phase2av" in report["blocked_actions"]


def test_phase2av_pretrain_gate_rejects_missing_descriptor_hash(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path, bad_hash=True)

    assert report["passed"] is False
    assert report["checks"]["descriptor_split_hashes_present"] is False
    assert "do_not_package_phase2av" in report["blocked_actions"]
