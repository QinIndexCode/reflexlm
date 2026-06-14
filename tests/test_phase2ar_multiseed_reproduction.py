import json
from pathlib import Path

from reflexlm.cli.audit_phase2ar_multiseed_reproduction import (
    audit_phase2ar_multiseed_reproduction,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _audit(primary_seed: int, reproduction_seed: int, *, passed: bool = True) -> dict:
    return {
        "passed": passed,
        "metrics": {
            "reproduction_success_rate": 1.0 if passed else 0.5,
            "row_count": 32,
            "training_contract": {
                "same_contract_except_seed_and_names": True,
                "split_hashes_match": True,
                "seed_changed": primary_seed != reproduction_seed,
                "primary_seed": primary_seed,
                "reproduction_seed": reproduction_seed,
            },
        },
    }


def test_phase2ar_multiseed_audit_accepts_three_same_model_seeds(tmp_path: Path) -> None:
    report = audit_phase2ar_multiseed_reproduction(
        reproduction_audit_jsons=[
            _write(tmp_path / "seed17.json", _audit(13, 17)),
            _write(tmp_path / "seed23.json", _audit(13, 23)),
        ]
    )

    assert report["passed"] is True
    assert report["metrics"]["unique_seeds"] == [13, 17, 23]
    assert "phase2ar_three_seed_same_model_reproduction_supported" in report["supported_claims"]
    assert "cross_model_reproduction" in report["unsupported_claims"]


def test_phase2ar_multiseed_audit_rejects_two_unique_seeds(tmp_path: Path) -> None:
    report = audit_phase2ar_multiseed_reproduction(
        reproduction_audit_jsons=[
            _write(tmp_path / "seed17.json", _audit(13, 17)),
        ]
    )

    assert report["passed"] is False
    assert report["checks"]["unique_seed_minimum_met"] is False


def test_phase2ar_multiseed_audit_rejects_contract_mismatch(tmp_path: Path) -> None:
    bad = _audit(13, 23)
    bad["metrics"]["training_contract"]["split_hashes_match"] = False
    report = audit_phase2ar_multiseed_reproduction(
        reproduction_audit_jsons=[
            _write(tmp_path / "seed17.json", _audit(13, 17)),
            _write(tmp_path / "seed23.json", bad),
        ]
    )

    assert report["passed"] is False
    assert report["checks"]["training_contracts_match_except_seed"] is False
