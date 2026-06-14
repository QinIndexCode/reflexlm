import json
from pathlib import Path

from reflexlm.cli.build_phase2an_candidate_artifact_controls import (
    build_phase2an_candidate_artifact_controls,
)


def _row(slot: int = 1) -> dict:
    return {
        "action_type": "RUN_COMMAND",
        "command": "repair_action_abc123def456",
        "command_slot": slot,
        "candidate_commands": [
            "repair_action_abc123def456 edit_scope=src/pkg/a.py target_symbol=fix_a",
            "repair_action_def456abc123 edit_scope=src/pkg/b.py target_symbol=fix_b",
        ],
        "nsi_reference": {
            "command_identity_slot:0": 2.0,
            "command_identity_slot:1": 6.0,
        },
        "source_trace": {"sealed_v3_used": False},
        "state_prompt": (
            "Header\n\n"
            "Candidate repair actions:\n"
            "- repair_action=repair_action_abc123def456; intent=apply_patch_and_rerun_tests; "
            "edit_scope=src/pkg/a.py; target_symbol=fix_a\n"
            "- repair_action=repair_action_def456abc123; intent=apply_patch_and_rerun_tests; "
            "edit_scope=src/pkg/b.py; target_symbol=fix_b\n\n"
            "Candidate commands:\n"
            "- repair_action_abc123def456 edit_scope=src/pkg/a.py target_symbol=fix_a\n"
            "- repair_action_def456abc123 edit_scope=src/pkg/b.py target_symbol=fix_b\n\n"
            "Head constraints:\n"
            "- RUN_COMMAND must select one repair action command slot."
        ),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_phase2an_controls_remove_candidate_text_artifacts(tmp_path: Path) -> None:
    rows = [_row(0), _row(1)]
    source = _write_jsonl(tmp_path / "source.jsonl", rows)
    manifest = build_phase2an_candidate_artifact_controls(
        original_jsonl=source,
        erased_jsonl=source,
        wrong_jsonl=source,
        output_dir=tmp_path / "out",
    )

    assert manifest["passed"] is True
    for output in manifest["outputs"].values():
        text = Path(output).read_text(encoding="utf-8")
        assert "repair_action_abc123def456" not in text
        assert "target_symbol=fix_a" not in text
        assert "bounded_repair_action" in text


def test_phase2an_controls_preserve_sidecar_scores(tmp_path: Path) -> None:
    source = _write_jsonl(tmp_path / "source.jsonl", [_row(1)])
    manifest = build_phase2an_candidate_artifact_controls(
        original_jsonl=source,
        erased_jsonl=source,
        wrong_jsonl=source,
        output_dir=tmp_path / "out",
    )
    row = json.loads(Path(manifest["outputs"]["neutral_original"]).read_text(encoding="utf-8"))

    assert row["nsi_reference"]["command_identity_slot:1"] == 6.0
    assert row["command_slot"] == 1
    assert len(row["candidate_commands"]) == 2
