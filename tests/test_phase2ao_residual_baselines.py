import json
from pathlib import Path

from reflexlm.cli.audit_phase2ao_residual_baselines import audit_phase2ao_residual_baselines


def _row(slot: int, count: int = 4) -> dict:
    return {
        "command_slot": slot,
        "candidate_commands": [f"candidate_{index}" for index in range(count)],
        "state_prompt": "visible state without useful overlap",
    }


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _eval(path: Path, accuracy: float) -> Path:
    path.write_text(
        json.dumps({"eval_metrics": {"command_slot_accuracy": accuracy}}),
        encoding="utf-8",
    )
    return path


def test_phase2ao_residual_baselines_accepts_model_above_priors(tmp_path: Path) -> None:
    train = _jsonl(tmp_path / "train.jsonl", [_row(0), _row(1), _row(2), _row(3)])
    eval_rows = [_row(0), _row(1), _row(2), _row(3)]
    report = audit_phase2ao_residual_baselines(
        train_jsonl=train,
        eval_jsonl=_jsonl(tmp_path / "eval.jsonl", eval_rows),
        erased_eval_json=_eval(tmp_path / "erased.json", 0.75),
    )

    assert report["passed"] is True
    assert report["checks"]["model_exceeds_max_nonleaky_baseline"] is True


def test_phase2ao_residual_baselines_rejects_model_explained_by_prior(tmp_path: Path) -> None:
    train = _jsonl(tmp_path / "train.jsonl", [_row(0), _row(0), _row(0), _row(1)])
    eval_rows = [_row(0), _row(0), _row(1), _row(2)]
    report = audit_phase2ao_residual_baselines(
        train_jsonl=train,
        eval_jsonl=_jsonl(tmp_path / "eval.jsonl", eval_rows),
        erased_eval_json=_eval(tmp_path / "erased.json", 0.5),
    )

    assert report["passed"] is False
    assert report["checks"]["model_exceeds_max_nonleaky_baseline"] is False
