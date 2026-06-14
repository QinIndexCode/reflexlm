from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PACKAGE_SOURCE = "package_internal_verification_cortex"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def audit_phase2cg_unified_package_live_repair(
    *,
    core_report_json: str | Path,
    live_rows_jsonl: str | Path,
    package_build_report_json: str | Path,
    native_head_probe_json: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    core = _read_json(core_report_json)
    rows = _read_jsonl(live_rows_jsonl)
    build = _read_json(package_build_report_json)
    probe = _read_json(native_head_probe_json)
    package = core.get("policy_metadata", {}).get("package_policy", {})
    visible_sources = {
        str(row.get("visible_control", {}).get("verification_source")) for row in rows
    }
    counterfactual_sources = {
        str(control.get("verification_source"))
        for row in rows
        for control in row.get("counterfactual_controls", {}).values()
        if isinstance(control, dict)
    }
    probe_commands = {str(row.get("selected_command")) for row in probe}
    probe_head_states = {
        json.dumps(row.get("heads"), sort_keys=True) for row in probe
    }
    checks = {
        "package_build_passed": build.get("passed") is True,
        "core_live_loop_passed": core.get("passed") is True,
        "minimum_live_rows_met": len(rows) >= 6,
        "minimum_live_repos_met": len({str(row.get("repo_origin")) for row in rows}) >= 3,
        "package_verification_cortex_packaged": package.get(
            "verification_cortex_packaged"
        )
        is True,
        "policy_declares_package_internal_verification": core.get(
            "policy_metadata", {}
        ).get("verification_control_source")
        == PACKAGE_SOURCE,
        "all_visible_decisions_package_internal": visible_sources == {PACKAGE_SOURCE},
        "all_counterfactual_decisions_package_internal": counterfactual_sources
        == {PACKAGE_SOURCE},
        "package_internal_verification_rate_complete": core.get("metrics", {}).get(
            "package_internal_verification_rate"
        )
        == 1.0,
        "visible_finish_complete": core.get("metrics", {}).get("visible_finish_rate")
        == 1.0,
        "erased_finish_zero": core.get("metrics", {}).get("erased_finish_rate") == 0.0,
        "wrong_finish_zero": core.get("metrics", {}).get("wrong_finish_rate") == 0.0,
        "frozen_finish_zero": core.get("metrics", {}).get("frozen_finish_rate") == 0.0,
        "actual_live_patch_execution_complete": core.get("metrics", {}).get(
            "live_patch_execution_success_rate"
        )
        == 1.0,
        "native_7b_head_probe_present": len(probe) >= 4,
        "native_7b_head_probe_is_receptor_insensitive": len(probe_commands) == 1
        and len(probe_head_states) == 1,
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2cg_unified_package_live_repair",
        "passed": passed,
        "ready_for_unified_package_multi_cortical_live_repair_claim": passed,
        "ready_for_monolithic_7b_native_head_verification_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            **core.get("metrics", {}),
            "visible_verification_sources": sorted(visible_sources),
            "counterfactual_verification_sources": sorted(counterfactual_sources),
            "native_head_probe_unique_commands": len(probe_commands),
            "native_head_probe_unique_head_states": len(probe_head_states),
        },
        "package_metadata": package,
        "supported_claims": [
            "one deployment package with distinct patch-selection and temporal verification cortical experts completed a bounded live public-repository patch-verify-stop loop without an external verification matcher"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "monolithic 7B native head verification",
            "arbitrary public-repository repair",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2ch_unified_package_long_run_stability_and_plasticity"
            if passed
            else "repair_phase2cg_unified_package_live_repair"
        ),
        "evidence": {
            "core_report_json": str(core_report_json),
            "live_rows_jsonl": str(live_rows_jsonl),
            "package_build_report_json": str(package_build_report_json),
            "native_head_probe_json": str(native_head_probe_json),
        },
    }
    output = Path(output_report_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the unified package multi-cortical live repair loop."
    )
    parser.add_argument("--core-report-json", required=True)
    parser.add_argument("--live-rows-jsonl", required=True)
    parser.add_argument("--package-build-report-json", required=True)
    parser.add_argument("--native-head-probe-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2cg_unified_package_live_repair(
        core_report_json=args.core_report_json,
        live_rows_jsonl=args.live_rows_jsonl,
        package_build_report_json=args.package_build_report_json,
        native_head_probe_json=args.native_head_probe_json,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
