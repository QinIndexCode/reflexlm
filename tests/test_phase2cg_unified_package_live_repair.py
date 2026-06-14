import json
from pathlib import Path

from reflexlm.cli.audit_phase2cg_unified_package_live_repair import (
    PACKAGE_SOURCE,
    audit_phase2cg_unified_package_live_repair,
)


def _write(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    return path


def test_phase2cg_audit_requires_package_internal_verification(tmp_path: Path) -> None:
    core = {
        "passed": True,
        "policy_metadata": {
            "verification_control_source": PACKAGE_SOURCE,
            "package_policy": {"verification_cortex_packaged": True},
        },
        "metrics": {
            "package_internal_verification_rate": 1.0,
            "visible_finish_rate": 1.0,
            "erased_finish_rate": 0.0,
            "wrong_finish_rate": 0.0,
            "frozen_finish_rate": 0.0,
            "live_patch_execution_success_rate": 1.0,
        },
    }
    rows = [
        {
            "repo_origin": f"repo-{index // 2}",
            "visible_control": {"verification_source": PACKAGE_SOURCE},
            "counterfactual_controls": {
                name: {"verification_source": PACKAGE_SOURCE}
                for name in ("erased_post", "wrong_post", "frozen_pre")
            },
        }
        for index in range(6)
    ]
    probe = [
        {"selected_command": "finish", "heads": {"stop_condition": 1}}
        for _ in range(4)
    ]
    report = audit_phase2cg_unified_package_live_repair(
        core_report_json=_write(tmp_path / "core.json", core),
        live_rows_jsonl=_write_jsonl(tmp_path / "rows.jsonl", rows),
        package_build_report_json=_write(tmp_path / "build.json", {"passed": True}),
        native_head_probe_json=_write(tmp_path / "probe.json", probe),
        output_report_json=tmp_path / "report.json",
    )

    assert report["passed"] is True
    assert report["ready_for_unified_package_multi_cortical_live_repair_claim"] is True
    assert report["ready_for_monolithic_7b_native_head_verification_claim"] is False


def test_phase2cg_audit_rejects_external_matcher_fallback(tmp_path: Path) -> None:
    core = {
        "passed": True,
        "policy_metadata": {
            "verification_control_source": "external_verification_matcher",
            "package_policy": {"verification_cortex_packaged": False},
        },
        "metrics": {
            "package_internal_verification_rate": 0.0,
            "visible_finish_rate": 1.0,
            "erased_finish_rate": 0.0,
            "wrong_finish_rate": 0.0,
            "frozen_finish_rate": 0.0,
            "live_patch_execution_success_rate": 1.0,
        },
    }
    rows = [
        {
            "repo_origin": f"repo-{index // 2}",
            "visible_control": {"verification_source": "external_verification_matcher"},
            "counterfactual_controls": {},
        }
        for index in range(6)
    ]
    report = audit_phase2cg_unified_package_live_repair(
        core_report_json=_write(tmp_path / "core.json", core),
        live_rows_jsonl=_write_jsonl(tmp_path / "rows.jsonl", rows),
        package_build_report_json=_write(tmp_path / "build.json", {"passed": True}),
        native_head_probe_json=_write(
            tmp_path / "probe.json",
            [{"selected_command": "finish", "heads": {"stop_condition": 1}}] * 4,
        ),
        output_report_json=tmp_path / "report.json",
    )

    assert report["passed"] is False
    assert report["checks"]["all_visible_decisions_package_internal"] is False
