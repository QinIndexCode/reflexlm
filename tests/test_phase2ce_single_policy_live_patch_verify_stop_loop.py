from pathlib import Path

from reflexlm.cli.run_phase2cd_public_repo_post_verification_control import (
    FINISH_INTENT,
)
from reflexlm.cli.run_phase2ce_single_policy_live_patch_verify_stop_loop import (
    ContinuousLiveRepairPolicy,
)
from reflexlm.models.semantic_matcher import _command_semantic_text


class _FakePackagePolicy:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def metadata(self) -> dict:
        return {"package_family": "phase2d_native_nervous_package"}


class _PackagedVerificationPolicy(_FakePackagePolicy):
    verification_cortex = object()

    def decide_verification(self, state):
        return {
            "selected_slot": 1,
            "selected_command": state.goal.command_allowlist[1],
            "scores": [0.0, 1.0],
            "package_internal_expert": True,
        }


class _VerificationMatcher:
    def score_state(self, state):
        return [
            1.0 if _command_semantic_text(command) == FINISH_INTENT else 0.0
            for command in state.goal.command_allowlist
        ]

    def metadata(self) -> dict:
        return {"matcher_family": "test_verification_matcher"}


def test_continuous_live_policy_keeps_one_lifecycle_for_stop_control(
    tmp_path: Path,
) -> None:
    package = _FakePackagePolicy()
    policy = ContinuousLiveRepairPolicy(
        package_policy=package,
        verification_matcher=_VerificationMatcher(),
        workspace_root=tmp_path,
    )
    policy.reset_episode("episode-1")
    policy.phase = "await_verification"

    decision = policy.decide_after_verification(
        pre_log={"exit_code": 1, "stdout": "FAILED test_example", "stderr": ""},
        post_log={"exit_code": 0, "stdout": "1 passed", "stderr": ""},
    )

    assert package.reset_count == 1
    assert decision["selected_intent"] == FINISH_INTENT
    assert decision["action_type"] == "DONE"
    assert policy.phase == "done"
    assert policy.metadata()["single_lifecycle"] is True


def test_continuous_live_policy_prefers_packaged_verification_cortex(
    tmp_path: Path,
) -> None:
    policy = ContinuousLiveRepairPolicy(
        package_policy=_PackagedVerificationPolicy(),
        verification_matcher=None,
        workspace_root=tmp_path,
    )
    policy.reset_episode("episode-1")
    policy.phase = "await_verification"

    decision = policy.decide_after_verification(
        pre_log={"exit_code": 1, "stdout": "FAILED test_example", "stderr": ""},
        post_log={"exit_code": 0, "stdout": "1 passed", "stderr": ""},
    )

    assert decision["finish_selected"] is True
    assert decision["verification_source"] == "package_internal_verification_cortex"
    assert decision["package_verification_decision"]["package_internal_expert"] is True
    assert policy.metadata()["verification_control_source"] == (
        "package_internal_verification_cortex"
    )
