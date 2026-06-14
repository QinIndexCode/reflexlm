import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_JSON = (
    ROOT
    / "artifacts"
    / "reports"
    / "phase2p_sealed_cross_model_transfer"
    / "phase2p_multiseed_cross_model_transfer_summary.json"
)
PAPER_DRAFT = ROOT / "paper_draft.md"


def test_phase2p_summary_records_preregistered_cross_model_gate() -> None:
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))

    assert summary["sealed_v3_used_for_training_sampling_or_tuning"] is False
    assert summary["passed"] is True
    assert summary["aggregate"]["model_count"] == 5
    assert summary["aggregate"]["seed_count"] == 3
    assert summary["aggregate"]["row_count"] == 15
    assert summary["aggregate"]["pass_rate"] == 1.0
    assert summary["aggregate"]["full_completion_min"] >= 0.85
    assert summary["aggregate"]["full_minus_no_nsi_min"] >= 0.15
    assert summary["aggregate"]["full_minus_native_head_only_min"] >= 0.10
    assert summary["aggregate"]["full_minus_continuation_only_min"] >= 0.15
    assert summary["aggregate"]["prompt_completion_max"] == 0.0
    assert summary["aggregate"]["react_completion_max"] == 0.0
    assert summary["aggregate"]["native_head_only_completion_max"] == 0.0
    assert summary["aggregate"]["continuation_only_completion_max"] == 0.0
    assert summary["aggregate"]["full_low_level_qwen_calls_max"] == 0
    assert summary["aggregate"]["full_qwen_on_non_debug_max"] == 0
    assert summary["aggregate"]["full_state_hallucination_max"] == 0.0


def test_phase2p_gate_rows_have_hashes_and_expected_models() -> None:
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    expected_models = {
        "qwen2_5_1_5b",
        "qwen2_5_3b",
        "qwen2_5_7b",
        "tinyllama_1_1b",
        "smollm2_360m",
    }

    rows = summary["rows"]

    assert {row["model_key"] for row in rows} == expected_models
    assert {row["seed"] for row in rows} == {13, 29, 47}
    assert all(row["passed"] for row in rows)
    assert all(len(row["gate_sha256"]) == 64 for row in rows)
    assert all((ROOT / row["gate_json"]).exists() for row in rows)


def test_phase2p_paper_boundary_is_updated_without_overclaiming() -> None:
    paper = PAPER_DRAFT.read_text(encoding="utf-8").lower()

    assert "phase2p" in paper
    assert "sealed cross-model transfer" in paper
    assert "not sealed cross-model" not in paper
    assert "has not yet shown sealed cross-model transfer" not in paper
    assert "epoch-making architecture status" in paper
    assert "production autonomy" in paper
    assert "still not proof of production autonomy" in paper or "still unsupported: production autonomy" in paper
