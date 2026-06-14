import json
from pathlib import Path

from reflexlm.cli.build_phase2am_natural_sidecar_controls import (
    build_phase2am_natural_sidecar_controls,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _row(slot: int, scores: list[float], prompt: str = "alpha") -> dict:
    return {
        "example_id": f"row-{slot}-{scores}",
        "state_prompt": prompt,
        "command_slot": slot,
        "candidate_commands": [
            "run alpha structural_probe_hash=a",
            "run beta structural_probe_hash=b",
        ],
        "nsi_reference": {
            "command_identity_slot:0": scores[0],
            "command_identity_slot:1": scores[1],
            "command_identity_slot:2": 0.0,
            "command_identity_slot:3": 0.0,
        },
        "source_trace": {"repo_id": "repo-a", "sealed_v3_used": False},
    }


def test_phase2am_controls_filter_sidecar_present_rows(tmp_path: Path) -> None:
    input_jsonl = _write_jsonl(
        tmp_path / "input.jsonl",
        [
            _row(0, [6.0, 0.0]),
            _row(1, [0.0, 6.0], prompt="beta"),
            _row(0, [0.0, 0.0]),
        ],
    )
    report = build_phase2am_natural_sidecar_controls(
        input_jsonl=input_jsonl,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
        min_rows=2,
        min_source_accuracy=0.0,
        max_source_accuracy=1.0,
    )

    assert report["passed"] is True
    assert report["row_count_input"] == 3
    assert report["row_count_selected"] == 2
    erased = [
        json.loads(line)
        for line in Path(report["splits"]["sidecar_erased"]["path"]).read_text().splitlines()
    ]
    assert all(row["nsi_reference"]["command_identity_confidence"] == 0.0 for row in erased)


def test_phase2am_controls_reject_source_overlap_ceiling(tmp_path: Path) -> None:
    input_jsonl = _write_jsonl(
        tmp_path / "input.jsonl",
        [_row(0, [6.0, 0.0], prompt="alpha") for _ in range(4)],
    )
    report = build_phase2am_natural_sidecar_controls(
        input_jsonl=input_jsonl,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
        min_rows=2,
        max_source_accuracy=0.75,
    )

    assert report["passed"] is False
    assert report["checks"]["source_overlap_not_ceiling"] is False
