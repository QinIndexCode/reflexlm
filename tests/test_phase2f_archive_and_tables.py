import json
from pathlib import Path

from reflexlm.cli.archive_phase2f_evidence import build_archive_manifest
from reflexlm.cli.build_phase2f_baseline_table import (
    build_table,
    default_external_trace_rows,
    markdown_table,
)
from reflexlm.cli.check_external_trace_gates import build_external_gate_report


def _write(path: Path, payload: dict | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _eval_payload(tmp_path: Path, name: str, completion: float) -> Path:
    run_path = tmp_path / "runs" / name
    run_path.mkdir(parents=True)
    (run_path / "trace_rows.jsonl").write_text(
        json.dumps(
            {
                "task_type": "test_failure_reflex",
                "policy_debug": {"action_source": "native_head_cortex"},
                "qwen_called": True,
                "cache_hit": False,
                "done": True,
                "reward": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_path / "run_manifest.json").write_text(json.dumps({"name": name}), encoding="utf-8")
    return _write(
        tmp_path / f"{name}.json",
        {
            "run_path": str(run_path),
            "episode_count": 10,
            "policy": {"json_text_target": False},
            "metrics": {
                "aggregate": {
                    "task_completion_rate": {"mean": completion, "positives": int(completion * 10)},
                    "model_calls": {"mean": 1.0},
                    "token_equivalent_cost": {"mean": 256.0},
                    "reaction_latency_ms": {"mean": 10.0},
                    "state_hallucination_rate": {"mean": 0.0},
                    "false_reflex_rate": {"mean": 0.0},
                }
            },
        },
    )


def test_phase2f_archive_manifest_hashes_are_deterministic(tmp_path: Path) -> None:
    report = tmp_path / "reports"
    package = tmp_path / "packages"
    paper = _write(tmp_path / "paper_draft.md", "# paper\n")
    _write(report / "phase2d_final_gate.json", {"passed": True})
    _write(
        report
        / "phase2f_rich_latent_fusion_canary_r16_alpha32_lr1e-4_len256_cap2048.training_summary.json",
        {"adapter": "x"},
    )
    eval_path = _eval_payload(report, "eval", 1.0)
    _write(package / "pkg" / "native_nervous_package.json", {"policy_label": "pkg"})

    first = build_archive_manifest(report_dir=report, package_root=package, paper_path=paper)
    second = build_archive_manifest(report_dir=report, package_root=package, paper_path=paper)

    assert first["aggregate_sha256"] == second["aggregate_sha256"]
    archived = {Path(row["source_path"]).name for row in first["files"]}
    assert "phase2d_final_gate.json" in archived
    assert eval_path.name in archived
    assert "run_manifest.json" in archived


def test_phase2f_baseline_table_uses_eval_json_metrics(tmp_path: Path) -> None:
    prompt = _eval_payload(tmp_path, "prompt", 0.2)
    full = _eval_payload(tmp_path, "full", 1.0)

    table = build_table(
        [
            ("prompt-only 7B", "external_trace_v1", prompt, "text_baseline"),
            ("Phase2F native package", "external_trace_v1", full, "strong_pass"),
        ]
    )
    markdown = markdown_table(table)

    assert table["rows"][0]["completion"] == 0.2
    assert table["rows"][1]["positives/episodes"] == "10/10"
    assert "Phase2F native package" in markdown


def test_external_trace_table_rows_accept_custom_adapter_name(tmp_path: Path) -> None:
    rows = default_external_trace_rows(
        tmp_path,
        dataset="external_trace_v3_semantic_required",
        adapter_name="phase2j_slotaware_full",
        native_package_label="Phase2J source-overlap-hard package",
    )

    assert rows[-1][0] == "Phase2J source-overlap-hard package"
    assert rows[-1][2] == tmp_path / "phase2j_slotaware_full.external_trace_v3_semantic_required_eval.json"
    assert rows[2][2] == tmp_path / "phase2j_slotaware_full.no_nsi_latent.external_trace_v3_semantic_required_eval.json"


def test_external_trace_gate_reports_single_mechanism_explanation(tmp_path: Path) -> None:
    full = _eval_payload(tmp_path, "full", 1.0)
    prompt = _eval_payload(tmp_path, "prompt", 0.1)
    react = _eval_payload(tmp_path, "react", 0.0)
    no_nsi = _eval_payload(tmp_path, "no_nsi", 0.5)
    native = _eval_payload(tmp_path, "native", 0.7)
    continuation = _eval_payload(tmp_path, "continuation", 1.0)
    manifest = _write(
        tmp_path / "manifest.json",
        {"sealed": True, "sealed_config_hash": "abc123"},
    )

    report = build_external_gate_report(
        full_eval_json=full,
        prompt_eval_json=prompt,
        react_eval_json=react,
        no_nsi_eval_json=no_nsi,
        native_head_only_eval_json=native,
        continuation_only_eval_json=continuation,
        dataset_manifest_json=manifest,
    )

    assert report["passed"] is True
    assert report["checks"]["mechanism_delta_or_explained"] is True
    assert report["claim_scope"] == "single_mechanism_explains_external_result"


def test_semantic_required_gate_requires_full_to_beat_continuation_only(tmp_path: Path) -> None:
    full = _eval_payload(tmp_path, "full", 1.0)
    prompt = _eval_payload(tmp_path, "prompt", 0.1)
    react = _eval_payload(tmp_path, "react", 0.0)
    no_nsi = _eval_payload(tmp_path, "no_nsi", 0.5)
    native = _eval_payload(tmp_path, "native", 0.7)
    continuation = _eval_payload(tmp_path, "continuation", 0.6)
    manifest = _write(
        tmp_path / "manifest.json",
        {
            "profile": "external_trace_v2_semantic_required",
            "sealed": True,
            "sealed_config_hash": "abc123",
        },
    )

    report = build_external_gate_report(
        full_eval_json=full,
        prompt_eval_json=prompt,
        react_eval_json=react,
        no_nsi_eval_json=no_nsi,
        native_head_only_eval_json=native,
        continuation_only_eval_json=continuation,
        dataset_manifest_json=manifest,
    )

    assert report["passed"] is True
    assert report["checks"]["beats_continuation_only"] is True
    assert report["claim_scope"] == "semantic_required_debug_cortex_supported"


def test_semantic_required_gate_fails_when_continuation_only_matches_full(tmp_path: Path) -> None:
    full = _eval_payload(tmp_path, "full", 1.0)
    prompt = _eval_payload(tmp_path, "prompt", 0.1)
    react = _eval_payload(tmp_path, "react", 0.0)
    no_nsi = _eval_payload(tmp_path, "no_nsi", 0.5)
    native = _eval_payload(tmp_path, "native", 0.7)
    continuation = _eval_payload(tmp_path, "continuation", 1.0)
    manifest = _write(
        tmp_path / "manifest.json",
        {
            "profile": "external_trace_v2_semantic_required",
            "sealed": True,
            "sealed_config_hash": "abc123",
        },
    )

    report = build_external_gate_report(
        full_eval_json=full,
        prompt_eval_json=prompt,
        react_eval_json=react,
        no_nsi_eval_json=no_nsi,
        native_head_only_eval_json=native,
        continuation_only_eval_json=continuation,
        dataset_manifest_json=manifest,
    )

    assert report["passed"] is False
    assert report["checks"]["beats_continuation_only"] is False
    assert report["claim_scope"] == "semantic_required_debug_cortex_not_proven"
