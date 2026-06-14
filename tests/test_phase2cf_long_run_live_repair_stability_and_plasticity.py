from reflexlm.cli.run_phase2cf_long_run_live_repair_stability_and_plasticity import (
    _build_phase2cf_report,
)


def _row(task_id: str, cycle: int, *, memory_hit: bool = True) -> dict:
    return {
        "task_id": f"{task_id}:cycle_{cycle + 1:02d}",
        "base_task_id": task_id,
        "cycle_index": cycle,
        "repo_origin": f"https://example.test/{task_id}.git",
        "selected_slot": 1,
        "selected_patch_correctly": True,
        "live_patch_execution_success": True,
        "actual_pre_test_failed": True,
        "actual_post_test_passed": True,
        "visible_control": {
            "finish_selected": True,
            "verification_source": "package_internal_verification_cortex",
        },
        "single_policy_lifecycle": True,
        "plasticity_feedback": {"accepted": True},
        "plasticity_prediction": {"memory_hit": memory_hit if cycle > 0 else False},
        "counterfactual_controls": {
            "erased_post": {
                "finish_selected": False,
                "verification_source": "package_internal_verification_cortex",
            },
            "wrong_post": {
                "finish_selected": False,
                "verification_source": "package_internal_verification_cortex",
            },
            "frozen_pre": {
                "finish_selected": False,
                "verification_source": "package_internal_verification_cortex",
            },
        },
        "lexical_visible_finish": False,
        "no_prior_visible_finish": False,
        "recorded_patch_artifact_used": False,
        "recorded_patch_artifact_used_for_fault_injection": True,
        "freeform_patch_generation": False,
        "sealed_feedback_used": False,
    }


def test_phase2cf_report_accepts_long_run_stability_and_plasticity() -> None:
    rows = [_row(f"task-{index}", cycle) for cycle in range(2) for index in range(6)]

    report = _build_phase2cf_report(
        live_rows=rows,
        cycles=2,
        train_rows=6,
        holdout_repos={f"https://example.test/task-{index}.git" for index in range(3)},
        tasks_jsonl="tasks.jsonl",
        head_records_jsonl="heads.jsonl",
        output_jsonl="rows.jsonl",
        policy_metadata={
            "package_policy": {
                "package_family": "phase2d_native_nervous_package",
                "verification_cortex_packaged": True,
            }
        },
        full_postflight_passed=True,
        require_package_internal_verification=True,
    )

    assert report["passed"] is True
    assert report["ready_for_long_run_live_repair_stability_and_plasticity_claim"] is True
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert report["metrics"]["stable_selection_rate"] == 1.0
    assert report["metrics"]["plasticity_memory_hit_rate_after_first_cycle"] == 1.0
    assert report["ready_for_unified_package_long_run_stability_and_plasticity_claim"] is True
    assert report["artifact_family"] == (
        "phase2ch_unified_package_long_run_stability_and_plasticity"
    )


def test_phase2cf_report_rejects_missing_plasticity_recall() -> None:
    rows = [
        _row(f"task-{index}", cycle, memory_hit=False)
        for cycle in range(2)
        for index in range(6)
    ]

    report = _build_phase2cf_report(
        live_rows=rows,
        cycles=2,
        train_rows=6,
        holdout_repos={f"https://example.test/task-{index}.git" for index in range(3)},
        tasks_jsonl="tasks.jsonl",
        head_records_jsonl="heads.jsonl",
        output_jsonl="rows.jsonl",
        policy_metadata={
            "package_policy": {"package_family": "phase2d_native_nervous_package"}
        },
        full_postflight_passed=True,
    )

    assert report["passed"] is False
    assert report["checks"]["plasticity_recall_after_first_cycle"] is False


def test_phase2cf_report_strict_mode_rejects_external_verification() -> None:
    rows = [_row(f"task-{index}", cycle) for cycle in range(2) for index in range(6)]
    for row in rows:
        row["visible_control"]["verification_source"] = "external_verification_matcher"

    report = _build_phase2cf_report(
        live_rows=rows,
        cycles=2,
        train_rows=6,
        holdout_repos={f"https://example.test/task-{index}.git" for index in range(3)},
        tasks_jsonl="tasks.jsonl",
        head_records_jsonl="heads.jsonl",
        output_jsonl="rows.jsonl",
        policy_metadata={
            "package_policy": {
                "package_family": "phase2d_native_nervous_package",
                "verification_cortex_packaged": True,
            }
        },
        full_postflight_passed=True,
        require_package_internal_verification=True,
    )

    assert report["passed"] is False
    assert report["checks"]["required_visible_verification_is_package_internal"] is False
