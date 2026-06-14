import torch

from reflexlm.cli.diagnose_phase2i_command_slots import (
    _zero_command_candidate_feature_groups_tensor,
    summarize_command_slot_records,
)
from reflexlm.llm.candidate_features import (
    CANDIDATE_FEATURE_DIM,
    COMMAND_IDENTITY_FEATURE_END,
    COMMAND_IDENTITY_FEATURE_START,
)


def test_command_slot_diagnostics_summarizes_sources_by_intent_and_slot() -> None:
    report = summarize_command_slot_records(
        [
            {
                "command_intent": "dependency_install",
                "gold_slot": 0,
                "predictions": {"slot_head": 0, "pairwise": 1, "source_overlap_baseline": 0},
            },
            {
                "command_intent": "test_rerun",
                "gold_slot": 2,
                "predictions": {"slot_head": 1, "pairwise": 2, "source_overlap_baseline": 2},
            },
        ]
    )

    assert report["command_record_count"] == 2
    assert report["sources"]["slot_head"]["accuracy"] == 0.5
    assert report["sources"]["pairwise"]["accuracy"] == 0.5
    assert report["sources"]["source_overlap_baseline"]["accuracy"] == 1.0
    assert report["sources"]["slot_head"]["by_intent"]["dependency_install"]["accuracy"] == 1.0
    assert report["sources"]["pairwise"]["by_gold_slot"]["2"]["predicted_slots"] == {2: 1}


def test_zero_command_candidate_feature_groups_tensor_removes_identity_only() -> None:
    features = torch.ones(2, 4, CANDIDATE_FEATURE_DIM)

    zeroed = _zero_command_candidate_feature_groups_tensor(features, ["candidate_identity"])

    identity_slice = slice(COMMAND_IDENTITY_FEATURE_START, COMMAND_IDENTITY_FEATURE_END)
    assert zeroed[:, :, identity_slice].abs().sum().item() == 0.0
    assert zeroed[:, :, :COMMAND_IDENTITY_FEATURE_START].abs().sum().item() > 0.0
    assert features[:, :, identity_slice].abs().sum().item() > 0.0
