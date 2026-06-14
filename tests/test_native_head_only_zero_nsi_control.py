import json
from pathlib import Path

from reflexlm.cli.build_native_head_only_zero_nsi_control import (
    CONTROL_OVERRIDE,
    build_native_head_only_zero_nsi_control,
)


def test_zero_nsi_control_erases_latent_without_changing_command(tmp_path: Path) -> None:
    source = tmp_path / "val.jsonl"
    row = {
        "command": "python -m pytest -q tests/test_a.py::test_one",
        "command_slot": 1,
        "candidate_commands": [
            "python -m pytest -q tests/test_a.py::test_zero",
            "python -m pytest -q tests/test_a.py::test_one",
        ],
        "nsi_reference": {"command_identity_slot:1": 4.0, "debug_action_stage": "source_inspected"},
        "runtime_overrides": ["debug_cortex_escalation"],
        "state_prompt": "runtime-visible text only",
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    output = tmp_path / "val.native.jsonl"
    manifest = tmp_path / "manifest.json"
    report = build_native_head_only_zero_nsi_control(
        source_jsonl=source,
        output_jsonl=output,
        output_json=manifest,
    )

    converted = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert converted["nsi_reference"] == {}
    assert CONTROL_OVERRIDE in converted["runtime_overrides"]
    assert converted["command"] == row["command"]
    assert converted["command_slot"] == row["command_slot"]
    assert converted["candidate_commands"] == row["candidate_commands"]
    assert converted["state_prompt"] == row["state_prompt"]
    assert report["rows"] == 1
    assert report["source_rows_with_nsi_reference"] == 1
    assert report["sealed_v3_used_for_training_or_tuning"] is False


def test_zero_nsi_control_is_idempotent_for_existing_override(tmp_path: Path) -> None:
    source = tmp_path / "val.jsonl"
    source.write_text(
        json.dumps(
            {
                "command": "pytest",
                "candidate_commands": ["pytest"],
                "nsi_reference": {},
                "runtime_overrides": [CONTROL_OVERRIDE],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "out.jsonl"
    build_native_head_only_zero_nsi_control(
        source_jsonl=source,
        output_jsonl=output,
    )

    converted = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert converted["runtime_overrides"].count(CONTROL_OVERRIDE) == 1
