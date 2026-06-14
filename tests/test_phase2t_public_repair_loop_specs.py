import json
from pathlib import Path

from reflexlm.cli.collect_phase2t_public_repair_loop_specs import (
    build_phase2t_public_repair_loop_spec_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SHIPPED_SPECS = ROOT / "docs" / "spec" / "phase2t_public_repair_loop_repo_specs.json"


TASK_FAMILIES = [
    "dependency_or_import_mismatch",
    "localized_unit_assertion",
    "stale_snapshot_update",
    "config_or_environment_marker",
    "multi_file_traceback_relation",
    "regression_after_partial_repair",
    "safety_blocked_command_temptation",
    "false_completion_trap",
]
FACTOR_LEVELS = {
    "candidate_count": [["2"], ["3"], ["4"], ["2", "3", "4"]],
    "evidence_density": [["low"], ["medium"], ["high"], ["low", "medium", "high"]],
    "repair_depth": [["one_edit"], ["two_edits"], ["stale_state_refresh"]],
    "failure_observability": [
        ["direct_traceback"],
        ["indirect_changed_file_relation"],
        ["ambiguous_same_intent_command"],
    ],
    "ambiguity_class": [
        ["same_intent_command"],
        ["same_file_read"],
        ["stage_transition"],
        ["patch_location_ambiguity"],
    ],
    "safety_pressure": [["none"], ["unsafe_command_lure"], ["rollback_required"]],
}
FULL_FACTOR_LEVELS = {
    factor: sorted({item for values in levels for item in values})
    for factor, levels in FACTOR_LEVELS.items()
}


def _write(path: Path, payload: list[dict]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _contract() -> dict:
    return {
        "sandbox_compatible": True,
        "source_repo_readonly": True,
        "patches_required": True,
        "tests_required": True,
        "rollback_required": True,
        "stop_required": True,
        "safety_pressure_included": True,
        "modern_baseline_measurable": True,
    }


def _provenance() -> dict:
    return {
        "public_repo": True,
        "license_metadata_present": True,
        "commit_pinned": True,
        "repo_origin_recorded": True,
    }


def _factor_payload(index: int) -> dict:
    return {
        factor: levels[index % len(levels)]
        for factor, levels in FACTOR_LEVELS.items()
    }


def _valid_specs() -> list[dict]:
    splits = ["train", "train", "train", "val", "val", "holdout", "holdout", "holdout"]
    specs: list[dict] = []
    for index, split in enumerate(splits):
        specs.append(
            {
                "repo_id": f"phase2t_repo_{index}",
                "split": split,
                "source_kind": "public_repo",
                "repo_url": f"https://github.com/example/phase2t-repo-{index}.git",
                "commit_hash": f"{index + 1:040x}",
                "license": "MIT",
                "provenance": _provenance(),
                "repair_loop_contract": _contract(),
                "task_families": TASK_FAMILIES,
                "factor_levels": FULL_FACTOR_LEVELS,
                "notes": "Public repair-loop candidate without gold labels or sealed feedback.",
            }
        )
    return specs


def test_phase2t_public_repair_loop_specs_accept_claim_bearing_collection_candidate(
    tmp_path: Path,
) -> None:
    report = build_phase2t_public_repair_loop_spec_manifest(
        repo_specs_json=_write(tmp_path / "specs.json", _valid_specs())
    )

    assert report["passed"] is True
    assert report["claim_bearing_collection_candidate"] is True
    assert report["claim_bearing_training_ready"] is False
    assert report["allowed_next_action"] == "run_phase2t_dynamic_repair_trace_collection"
    assert report["rollups"]["missing_task_families"] == []
    assert report["rollups"]["missing_factor_levels"] == {}
    assert report["rollups"]["split_missing_task_families"] == {
        "holdout": [],
        "train": [],
        "val": [],
    }
    assert report["checks"]["repo_origins_split_disjoint"] is True


def test_phase2t_shipped_public_repair_loop_specs_pass_spec_gate() -> None:
    report = build_phase2t_public_repair_loop_spec_manifest(repo_specs_json=SHIPPED_SPECS)

    assert report["passed"] is True
    assert report["allowed_next_action"] == "run_phase2t_dynamic_repair_trace_collection"
    assert report["claim_bearing_training_ready"] is False
    assert report["rollups"]["missing_task_families"] == []
    assert report["rollups"]["missing_factor_levels"] == {}
    assert report["rollups"]["split_missing_task_families"] == {
        "holdout": [],
        "train": [],
        "val": [],
    }
    assert report["checks"]["repo_origins_unique"] is True


def test_phase2t_public_repair_loop_specs_reject_sealed_gold_candidate_markers(
    tmp_path: Path,
) -> None:
    specs = _valid_specs()
    specs[0]["notes"] = "external_trace_v3_semantic_required candidate_0 expected_patch"

    report = build_phase2t_public_repair_loop_spec_manifest(
        repo_specs_json=_write(tmp_path / "specs.json", specs)
    )

    assert report["passed"] is False
    assert "remove_sealed_gold_candidate_or_expected_patch_markers" in report[
        "blocked_actions"
    ]
    assert report["repos"][0]["checks"]["no_forbidden_or_sealed_markers"] is False


def test_phase2t_public_repair_loop_specs_reject_repo_reuse_across_splits(
    tmp_path: Path,
) -> None:
    specs = _valid_specs()
    specs[4]["repo_url"] = specs[0]["repo_url"]

    report = build_phase2t_public_repair_loop_spec_manifest(
        repo_specs_json=_write(tmp_path / "specs.json", specs)
    )

    assert report["passed"] is False
    assert "do_not_reuse_repo_origin_across_phase2t_splits" in report["blocked_actions"]


def test_phase2t_public_repair_loop_specs_reject_missing_contract_and_provenance(
    tmp_path: Path,
) -> None:
    specs = _valid_specs()
    specs[0]["commit_hash"] = "not-pinned"
    specs[0]["license"] = ""
    specs[0]["provenance"]["commit_pinned"] = False
    specs[0]["repair_loop_contract"]["rollback_required"] = False

    report = build_phase2t_public_repair_loop_spec_manifest(
        repo_specs_json=_write(tmp_path / "specs.json", specs)
    )

    assert report["passed"] is False
    assert "revise_phase2t_repo_specs_shape_or_contract" in report["blocked_actions"]
    repo_checks = report["repos"][0]["checks"]
    assert repo_checks["commit_pinned"] is False
    assert repo_checks["license_metadata_present"] is False
    assert repo_checks["provenance_flags_present"] is False
    assert repo_checks["repair_loop_contract_present"] is False


def test_phase2t_public_repair_loop_specs_reject_weak_pressure_coverage(
    tmp_path: Path,
) -> None:
    specs = _valid_specs()[:3]
    specs[0]["task_families"] = [TASK_FAMILIES[0]]
    specs[0]["factor_levels"] = _factor_payload(0)

    report = build_phase2t_public_repair_loop_spec_manifest(
        repo_specs_json=_write(tmp_path / "specs.json", specs)
    )

    assert report["passed"] is False
    assert "add_more_repo_origin_disjoint_phase2t_specs" in report["blocked_actions"]
    assert "add_phase2t_splitwise_task_family_coverage_before_collection" in report[
        "blocked_actions"
    ]
    assert "add_phase2t_splitwise_graded_factor_coverage_before_collection" in report[
        "blocked_actions"
    ]
