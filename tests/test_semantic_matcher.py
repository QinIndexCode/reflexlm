import re
from types import SimpleNamespace

import torch
from torch import nn

from reflexlm.models.semantic_matcher import (
    CausalLMConditionalSemanticMatcher,
    HashedDualEncoderSemanticMatcher,
    RecencyWeightedSemanticMatcher,
    _command_semantic_text,
    _receptor_text,
)
from reflexlm.schema import (
    FileSystemState,
    GoalSpec,
    ProcessState,
    RuntimeEvidenceState,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
)


def _groups() -> dict[str, dict[str, list[str]]]:
    return {
        "dependency": {
            "observations": [
                "module import failed",
                "required package is unavailable",
            ],
            "commands": [
                "install required package",
                "restore dependency",
            ],
        },
        "permission": {
            "observations": [
                "write access denied",
                "operation is not permitted",
            ],
            "commands": [
                "repair file permissions",
                "grant write access",
            ],
        },
        "port": {
            "observations": [
                "network address is already in use",
                "listener cannot bind to occupied port",
            ],
            "commands": [
                "release occupied port",
                "stop existing listener",
            ],
        },
    }


def test_dual_encoder_learns_observation_command_compatibility() -> None:
    matcher = HashedDualEncoderSemanticMatcher(bins=256, embedding_dim=32, seed=7)
    summary = matcher.fit(_groups(), epochs=300, learning_rate=0.04)

    scores = matcher.score_texts(
        "operation is not permitted for this file",
        ["release occupied port", "repair file permissions", "install package"],
    )

    assert summary.training_top1_accuracy == 1.0
    assert scores.index(max(scores)) == 1
    assert matcher.metadata()["runtime_ontology_lookup"] is False


def test_dual_encoder_round_trips_checkpoint(tmp_path) -> None:
    matcher = HashedDualEncoderSemanticMatcher(bins=256, embedding_dim=32, seed=11)
    matcher.fit(_groups(), epochs=250, learning_rate=0.04)
    before = matcher.score_texts("module import failed", ["repair permissions", "install package"])

    loaded = HashedDualEncoderSemanticMatcher.load(matcher.save(tmp_path / "matcher.pt"))
    after = loaded.score_texts("module import failed", ["repair permissions", "install package"])

    assert after == before


def test_lexical_residual_supports_unseen_vocabulary() -> None:
    matcher = HashedDualEncoderSemanticMatcher(
        bins=256,
        embedding_dim=32,
        seed=5,
        lexical_residual_weight=3.0,
    )
    matcher.fit(_groups(), epochs=250, learning_rate=0.04)

    scores = matcher.score_texts(
        "TLS certificate expired during handshake",
        ["free disk space", "renew expired TLS certificate", "release database lock"],
    )

    assert scores.index(max(scores)) == 1


def test_lexical_residual_handles_generic_inflection() -> None:
    matcher = HashedDualEncoderSemanticMatcher(
        bins=256,
        embedding_dim=32,
        seed=19,
        lexical_residual_weight=3.0,
    )
    matcher.fit(_groups(), epochs=250, learning_rate=0.04)

    scores = matcher.score_texts(
        "database is locked by another process",
        ["renew expired certificate", "reduce memory use", "release database lock"],
    )

    assert scores.index(max(scores)) == 2


class _TinyTokenizer:
    def __init__(self) -> None:
        words = [
            "<bos>",
            "storage",
            "purge",
            "rotate",
            "obsolete",
            "cache",
            "files",
            "certificate",
        ]
        self.vocab = {word: index for index, word in enumerate(words)}

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int,
    ) -> dict[str, list[int]]:
        del truncation
        ids = [
            self.vocab.get(word, len(self.vocab))
            for word in re.findall(r"[a-z]+", text.lower())
        ]
        if add_special_tokens:
            ids.insert(0, self.vocab["<bos>"])
        return {"input_ids": ids[-max_length:]}


class _TinyConditionalModel(nn.Module):
    def __init__(self, tokenizer: _TinyTokenizer) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.storage_id = tokenizer.vocab["storage"]
        self.purge_id = tokenizer.vocab["purge"]
        self.vocab_size = len(tokenizer.vocab) + 1

    def forward(self, *, input_ids: torch.Tensor) -> SimpleNamespace:
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab_size, device=input_ids.device)
        for position in range(length):
            prefix = input_ids[:, : position + 1]
            contains_storage = (prefix == self.storage_id).any(dim=1)
            logits[contains_storage, position, self.purge_id] = 8.0
        return SimpleNamespace(logits=logits)


def test_causal_matcher_uses_conditional_association_not_command_prior() -> None:
    tokenizer = _TinyTokenizer()
    matcher = CausalLMConditionalSemanticMatcher(
        model=_TinyConditionalModel(tokenizer),
        tokenizer=tokenizer,
        model_name="tiny-test-cortex",
    )

    scores = matcher.score_texts(
        "storage volume exhausted",
        ["rotate certificate", "purge obsolete cache files"],
    )

    assert scores.index(max(scores)) == 1
    assert matcher.metadata()["runtime_ontology_lookup"] is False
    assert matcher.metadata()["free_form_action_generation"] is False


def test_command_semantic_text_removes_executable_wrapper() -> None:
    text = _command_semantic_text(
        r"C:\work\.venv\Scripts\python.exe -c "
        "\"print('reduce memory use: phase2ca_repo_candidate')\""
    )

    assert text == "reduce memory use"
    assert _command_semantic_text("python -c \"print('rotate TLS certificate')\"") == (
        "rotate TLS certificate"
    )


def test_command_semantic_text_prefers_explicit_bounded_intent() -> None:
    command = (
        'python -c "import base64; print(base64.b64decode(\'opaque\'))" '
        'opaque_payload --intent "restore missing method"'
    )

    assert _command_semantic_text(command) == "restore missing method"
    assert _command_semantic_text("runner --intent=verify_repair opaque") == (
        "verify_repair"
    )


def test_receptor_text_discards_empty_channel_boundaries() -> None:
    state = SystemStateFrame(
        time=TimeState(),
        goal=GoalSpec(task_type=TaskType.TEST_FAILURE, description="generic goal"),
        process=ProcessState(),
        terminal=TerminalState(),
        filesystem=FileSystemState(),
        runtime_evidence=RuntimeEvidenceState(
            terminal_observations=["storage volume exhausted"]
        ),
    )

    assert _receptor_text(state) == "storage volume exhausted"


def test_recency_weighted_matcher_preserves_history_but_prioritizes_latest_frame() -> None:
    class _FrameMatcher:
        def score_texts(self, observation: str, commands: list[str]) -> list[float]:
            del commands
            if "latest import failure" in observation:
                return [0.0, 1.0]
            if "older attribute failure" in observation:
                return [1.0, 0.0]
            return [0.0, 0.0]

    state = SystemStateFrame(
        time=TimeState(),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description="adapt to latest receptor",
            command_allowlist=[
                "runner --intent 'restore missing method'",
                "runner --intent 'load required library'",
            ],
        ),
        process=ProcessState(),
        terminal=TerminalState(),
        filesystem=FileSystemState(),
        runtime_evidence=RuntimeEvidenceState(
            terminal_observations=[
                "older attribute failure",
                "latest import failure",
            ]
        ),
    )

    scores = RecencyWeightedSemanticMatcher(
        _FrameMatcher(),
        recency_decay=0.25,
    ).score_state(state)

    assert scores[1] > scores[0]
