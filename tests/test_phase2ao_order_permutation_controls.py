import json
from pathlib import Path

from reflexlm.cli.build_phase2ao_order_permutation_controls import (
    build_phase2ao_order_permutation_controls,
    permute_candidate_order,
)


def _row(slot: int = 1) -> dict:
    return {
        "example_id": "ex-1",
        "episode_id": "ep-1",
        "command_slot": slot,
        "candidate_commands": ["a", "b", "c"],
        "nsi_reference": {
            "command_identity_slot:0": 1.0,
            "command_identity_slot:1": 6.0,
            "command_identity_slot:2": 2.0,
            "command_identity_slot:3": 0.0,
        },
        "source_trace": {"sealed_v3_used": False},
        "state_prompt": (
            "Header\n\n"
            "Candidate repair actions:\n"
            "- repair_action=bounded_repair_action; intent=apply_patch_and_rerun_tests; "
            "edit_scope=bounded_public_source_patch; target_symbol=runtime_visible_symbol\n"
            "- repair_action=bounded_repair_action; intent=apply_patch_and_rerun_tests; "
            "edit_scope=bounded_public_source_patch; target_symbol=runtime_visible_symbol\n"
            "- repair_action=bounded_repair_action; intent=apply_patch_and_rerun_tests; "
            "edit_scope=bounded_public_source_patch; target_symbol=runtime_visible_symbol\n\n"
            "Candidate commands:\n"
            "- bounded_repair_action intent=apply_patch_and_rerun_tests "
            "edit_scope=bounded_public_source_patch target_symbol=runtime_visible_symbol\n"
            "- bounded_repair_action intent=apply_patch_and_rerun_tests "
            "edit_scope=bounded_public_source_patch target_symbol=runtime_visible_symbol\n"
            "- bounded_repair_action intent=apply_patch_and_rerun_tests "
            "edit_scope=bounded_public_source_patch target_symbol=runtime_visible_symbol\n\n"
            "Head constraints:\n"
            "- RUN_COMMAND must select one repair action command slot."
        ),
    }


def test_phase2ao_permutation_remaps_gold_and_sidecar() -> None:
    row = _row(slot=1)
    out = permute_candidate_order(row, seed=7)
    mapping = out["phase2ao_order_permutation"]["new_to_old"]
    old_gold = row["command_slot"]
    new_gold = out["command_slot"]

    assert mapping[str(new_gold)] == old_gold
    assert out["nsi_reference"][f"command_identity_slot:{new_gold}"] == 6.0
    assert out["candidate_commands"][new_gold] == row["candidate_commands"][old_gold]


def test_phase2ao_builds_permuted_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "in.jsonl"
    input_path.write_text("\n".join(json.dumps(_row(slot=1)) for _ in range(2)) + "\n")
    report = build_phase2ao_order_permutation_controls(
        input_jsonl=input_path,
        output_jsonl=tmp_path / "out.jsonl",
    )

    assert report["passed"] is True
    assert report["non_identity_permutation_rows"] == 2
