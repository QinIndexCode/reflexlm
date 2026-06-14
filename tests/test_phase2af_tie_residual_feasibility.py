import json
from pathlib import Path

from reflexlm.cli.audit_phase2af_tie_residual_feasibility import (
    audit_phase2af_tie_residual_feasibility,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(
    *,
    command_slot: int,
    identity_margin: float,
    state_prompt: str,
    candidates: list[str],
) -> dict:
    return {
        "example_id": "case",
        "source_trace": {"repo_id": "repo"},
        "state_prompt": state_prompt,
        "candidate_commands": candidates,
        "command_slot": command_slot,
        "nsi_reference": {"command_identity_margin": identity_margin},
    }


def test_phase2af_tie_residual_feasibility_rejects_unresolved_identity_tie(
    tmp_path: Path,
) -> None:
    path = _write_jsonl(
        tmp_path / "heads.jsonl",
        [
            _row(
                command_slot=1,
                identity_margin=0.0,
                state_prompt="Failure has no candidate-specific cue.",
                candidates=[
                    "repair_a edit_scope=bounded_public_source_patch target_symbol=hash_a",
                    "repair_b edit_scope=bounded_public_source_patch target_symbol=hash_b",
                ],
            )
        ],
    )

    report = audit_phase2af_tie_residual_feasibility(
        head_jsonl=path,
        split_name="holdout",
        output_json=tmp_path / "audit.json",
    )

    assert report["passed"] is False
    assert report["metrics"]["identity_tie_rows"] == 1
    assert report["metrics"]["unresolved_identity_tie_rows"] == 1
    assert report["failure_distribution"]["unresolved_by_repo"] == {"repo": 1}
    assert "do_not_package_phase2af" in report["blocked_actions"]


def test_phase2af_tie_residual_feasibility_accepts_visible_unique_cue(
    tmp_path: Path,
) -> None:
    path = _write_jsonl(
        tmp_path / "heads.jsonl",
        [
            _row(
                command_slot=1,
                identity_margin=0.0,
                state_prompt="Runtime evidence references pkg/b.py and symbol target_b.",
                candidates=[
                    "repair_a edit_scope=pkg/a.py target_symbol=target_a",
                    "repair_b edit_scope=pkg/b.py target_symbol=target_b",
                ],
            )
        ],
    )

    report = audit_phase2af_tie_residual_feasibility(
        head_jsonl=path,
        split_name="holdout",
        output_json=tmp_path / "audit.json",
    )

    assert report["passed"] is True
    assert report["metrics"]["identity_tie_rows"] == 1
    assert report["metrics"]["source_disambiguated_identity_tie_correct_rows"] == 1
    assert report["blocked_actions"] == []


def test_phase2af_tie_residual_feasibility_ignores_non_tie_rows(
    tmp_path: Path,
) -> None:
    path = _write_jsonl(
        tmp_path / "heads.jsonl",
        [
            _row(
                command_slot=0,
                identity_margin=3.0,
                state_prompt="No extra source cue needed because identity sidecar is unique.",
                candidates=[
                    "repair_a edit_scope=pkg/a.py target_symbol=target_a",
                    "repair_b edit_scope=pkg/b.py target_symbol=target_b",
                ],
            )
        ],
    )

    report = audit_phase2af_tie_residual_feasibility(
        head_jsonl=path,
        split_name="holdout",
        output_json=tmp_path / "audit.json",
    )

    assert report["passed"] is True
    assert report["metrics"]["identity_tie_rows"] == 0
    assert report["metrics"]["unresolved_identity_tie_rows"] == 0
