import json
from pathlib import Path

from reflexlm.cli.build_phase2ag_sidecar_controls import (
    build_phase2ag_sidecar_controls,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _row(slot: int = 0) -> dict:
    return {
        "example_id": "e1",
        "command_slot": slot,
        "candidate_commands": ["a", "b", "c"],
        "nsi_reference": {
            "command_identity_confidence": 18.0,
            "command_identity_margin": 16.0,
            "command_identity_slot:0": 18.0,
            "command_identity_slot:1": 2.0,
            "command_identity_slot:2": 0.0,
            "command_identity_slot:3": 0.0,
        },
    }


def test_phase2ag_sidecar_controls_erases_and_misaligns_scores(tmp_path: Path) -> None:
    input_jsonl = _write_jsonl(tmp_path / "input.jsonl", [_row()])
    report = build_phase2ag_sidecar_controls(
        input_jsonl=input_jsonl,
        output_dir=tmp_path / "controls",
        manifest_json=tmp_path / "manifest.json",
    )

    assert report["passed"] is True
    erased = json.loads(
        Path(report["controls"]["sidecar_erased"]["path"]).read_text(encoding="utf-8").splitlines()[0]
    )
    wrong = json.loads(
        Path(report["controls"]["wrong_sidecar"]["path"]).read_text(encoding="utf-8").splitlines()[0]
    )
    assert erased["nsi_reference"]["command_identity_confidence"] == 0.0
    assert erased["nsi_reference"]["command_identity_slot:0"] == 0.0
    assert wrong["nsi_reference"]["command_identity_slot:1"] == 18.0
    assert wrong["command_slot"] == 0


def test_phase2ag_sidecar_controls_rejects_empty_input(tmp_path: Path) -> None:
    input_jsonl = _write_jsonl(tmp_path / "input.jsonl", [])

    report = build_phase2ag_sidecar_controls(
        input_jsonl=input_jsonl,
        output_dir=tmp_path / "controls",
        manifest_json=tmp_path / "manifest.json",
    )

    assert report["passed"] is False
    assert "do_not_claim_sidecar_dependence" in report["blocked_actions"]
