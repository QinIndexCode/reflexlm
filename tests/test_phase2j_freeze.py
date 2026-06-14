import json
from pathlib import Path

from reflexlm.cli.freeze_phase2j_stagegain import build_phase2j_stagegain_freeze_manifest


def _write(path: Path, payload: dict | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _eval_payload(tmp_path: Path, name: str, completion: float) -> dict:
    run_path = tmp_path / "runs" / name
    _write(run_path / "trace_rows.jsonl", "")
    return {
        "episode_count": 64,
        "run_path": str(run_path),
        "metrics": {
            "aggregate": {
                "task_completion_rate": {
                    "mean": completion,
                    "positives": round(completion * 64),
                    "count": 64,
                },
                "state_hallucination_rate": {"mean": 0.0, "count": 64},
                "model_calls": {"mean": 1.0, "count": 64},
                "oracle_step_accuracy": {"mean": completion, "count": 64},
            }
        },
    }


def test_phase2j_stagegain_freeze_manifest_records_claim_boundary(tmp_path: Path) -> None:
    report = tmp_path / "reports"
    actiongate = tmp_path / "actiongate"
    package_root = tmp_path / "packages"
    for name, completion in {
        "full": 1.0,
        "no_nsi_latent": 0.28125,
        "native_head_only": 1.0,
        "continuation_only": 0.0,
        "prompt_only": 0.0,
        "react": 0.0,
    }.items():
        _write(report / f"p2j_stagegain.{name}.external_trace_v3_eval.json", _eval_payload(tmp_path, name, completion))
    _write(actiongate / "p2j_actionbal_stagegain_full_data_health.json", {"passed": True})
    _write(actiongate / "p2j_actionbal_stagegain_full_pretrain_gate.json", {"passed": True})
    _write(actiongate / "p2j_actionbal_stagegain_full.training_summary.json", {"ok": True})
    _write(actiongate / "p2j_actionbal_stagegain_full_postflight.json", {"passed": True, "metrics": {"model_minus_source_overlap_accuracy": 0.8}})
    _write(report / "p2j_stagegain_external_v3_exact_baseline_table.json", {"rows": []})
    _write(report / "p2j_stagegain_external_v3_exact_baseline_table.md", "| a |\n")
    _write(report / "p2j_stagegain_external_v3_gate.json", {"passed": True})
    _write(report / "p2j_stagegain_external_v3_strict_delta_review.json", {"strict_full_package_gate_passed": False})
    _write(report / "p2j_stagegain_reproducibility_report.md", "# report\n")
    for package in [
        "p2j_actionbal_stagegain_full1024_val288",
        "p2j_actionbal_stagegain_full1024_val288_no_nsi_latent",
        "p2j_actionbal_stagegain_full1024_val288_native_head_only",
        "p2j_actionbal_stagegain_full1024_val288_continuation_only",
    ]:
        _write(package_root / package / "native_nervous_package.json", {"package": package})

    manifest = build_phase2j_stagegain_freeze_manifest(
        report_dir=report,
        actiongate_report_dir=actiongate,
        package_root=package_root,
    )

    assert manifest["frozen"] is True
    assert manifest["sealed_v3_used_for_training_or_tuning"] is False
    assert manifest["checks"]["full_beats_no_nsi_by_15pp"] is True
    assert manifest["checks"]["full_does_not_beat_native_head_only_by_10pp"] is True
    assert "full package beats native-head-only" in manifest["unsupported_claims"]
    assert manifest["metrics"]["deltas"]["full_minus_native_head_only"] == 0.0
