import json
from pathlib import Path

from reflexlm.cli.audit_phase2i_latent_necessity import build_phase2i_latent_necessity_audit
from reflexlm.cli.generate_debug_cortex_challenge import build_debug_cortex_challenge


def _write_head_rows(path: Path, *, latent: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_prompt = "\n".join(
        [
            "Phase 2C native nervous interface state input.",
            "",
            "Visible transition summary:",
            f"failure_signal={'latent_required' if latent else 'other'}",
            "source_inspected=True",
            "last_command_intent=test_rerun",
            "candidate_command_intents=0:test_rerun,1:test_rerun,2:test_rerun",
            "",
            "Receptor state:",
            "goal_description=React to a failed test run using structured actions.",
            (
                "stdout_delta=<compressed_failure_signal>"
                if latent
                else "stdout_delta=Source inspected: rerun tests/alpha/test_a.py."
            ),
            "stderr_delta=<compressed_failure_signal>" if latent else "stderr_delta=",
            "last_command=python -m pytest -q tests/alpha/test_a.py",
            "",
            "Candidate commands:",
            "- python -m pytest -q tests/alpha/test_a.py",
            "- python -m pytest -q tests/beta/test_b.py",
            "- python -m pytest -q tests/gamma/test_c.py",
            "",
            "Candidate files:",
            "- src/example.py",
        ]
    )
    row = {
        "example_id": "synthetic-latent:2",
        "state_prompt": state_prompt,
        "candidate_commands": [
            "python -m pytest -q tests/alpha/test_a.py",
            "python -m pytest -q tests/beta/test_b.py",
            "python -m pytest -q tests/gamma/test_c.py",
        ],
        "command_slot": 1,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def test_phase2i_latent_necessity_audit_accepts_nonsealed_latent_probe_with_command_identity(
    tmp_path: Path,
) -> None:
    challenge_dir = tmp_path / "latent_sensitive"
    build_debug_cortex_challenge(
        challenge_dir,
        profile="phase2f_latent_sensitive",
        episodes_per_scenario=2,
    )
    train = _write_head_rows(tmp_path / "heads" / "train.jsonl")
    val = _write_head_rows(tmp_path / "heads" / "val.jsonl")

    report = build_phase2i_latent_necessity_audit(
        head_splits={"train": train, "val": val},
        challenge_splits={"phase2f_latent_sensitive": challenge_dir / "challenge.jsonl"},
        nsi_latent_fields=("salience", "reflex_command_hash"),
        min_latent_command_rows=1,
        min_latent_same_intent_rate=0.20,
    )

    assert report["passed"] is True
    assert report["checks"]["non_sealed_inputs_only"] is True
    assert report["checks"]["latent_challenge_source_overlap_not_sufficient"] is True
    assert report["checks"]["head_latent_coverage"] is True
    assert report["checks"]["nsi_latent_command_identity_available"] is True


def test_phase2i_latent_necessity_audit_rejects_same_intent_without_command_identity(
    tmp_path: Path,
) -> None:
    challenge_dir = tmp_path / "latent_sensitive"
    build_debug_cortex_challenge(
        challenge_dir,
        profile="phase2f_latent_sensitive",
        episodes_per_scenario=2,
    )
    train = _write_head_rows(tmp_path / "heads" / "train.jsonl")
    val = _write_head_rows(tmp_path / "heads" / "val.jsonl")

    report = build_phase2i_latent_necessity_audit(
        head_splits={"train": train, "val": val},
        challenge_splits={"phase2f_latent_sensitive": challenge_dir / "challenge.jsonl"},
        min_latent_command_rows=1,
        min_latent_same_intent_rate=0.20,
    )

    assert report["passed"] is False
    assert report["checks"]["nsi_latent_command_identity_available"] is False
    assert "command or slot identity" in report["interpretation"]["architecture_blocker"]


def test_phase2i_latent_necessity_audit_rejects_source_overlap_semantic_probe(
    tmp_path: Path,
) -> None:
    challenge_dir = tmp_path / "semantic_val"
    build_debug_cortex_challenge(
        challenge_dir,
        profile="phase2i_semantic_val",
        episodes_per_scenario=1,
    )

    report = build_phase2i_latent_necessity_audit(
        head_splits={},
        challenge_splits={"phase2i_semantic_val": challenge_dir / "challenge.jsonl"},
        min_latent_command_rows=1,
        require_head_coverage=False,
    )

    assert report["passed"] is False
    assert report["checks"]["latent_challenge_present"] is False
    assert report["challenge_rollup"]["source_overlap_accuracy"] == 1.0


def test_phase2i_latent_necessity_audit_rejects_head_split_without_latent_rows(
    tmp_path: Path,
) -> None:
    challenge_dir = tmp_path / "latent_sensitive"
    build_debug_cortex_challenge(
        challenge_dir,
        profile="phase2f_latent_sensitive",
        episodes_per_scenario=2,
    )
    train = _write_head_rows(tmp_path / "heads" / "train.jsonl", latent=False)
    val = _write_head_rows(tmp_path / "heads" / "val.jsonl", latent=False)

    report = build_phase2i_latent_necessity_audit(
        head_splits={"train": train, "val": val},
        challenge_splits={"phase2f_latent_sensitive": challenge_dir / "challenge.jsonl"},
        min_latent_command_rows=1,
    )

    assert report["passed"] is False
    assert report["checks"]["head_latent_coverage"] is False
