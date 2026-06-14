from __future__ import annotations

import re
import json
from collections.abc import Iterable
from typing import Any

from reflexlm.models.features import MAX_CANDIDATE_SLOTS


COMMAND_INTENT_ORDER = (
    "dependency_install",
    "snapshot_update",
    "test_rerun",
    "other",
)
COMMAND_INTENT_COUNT = len(COMMAND_INTENT_ORDER)
PAIRWISE_COMMAND_POLICIES = ("all", "ambiguous_intent")
COMMAND_CANDIDATE_BASE_FEATURE_DIM = 15
SOURCE_EVIDENCE_FEATURE_DIM = 5
COMMAND_SLOT_POSITION_FEATURE_DIM = MAX_CANDIDATE_SLOTS
COMMAND_IDENTITY_CANDIDATE_FEATURE_DIM = 4
COMMAND_INTENT_FEATURE_START = COMMAND_CANDIDATE_BASE_FEATURE_DIM
COMMAND_INTENT_FEATURE_END = COMMAND_INTENT_FEATURE_START + COMMAND_INTENT_COUNT
SOURCE_EVIDENCE_FEATURE_START = COMMAND_INTENT_FEATURE_END
SOURCE_EVIDENCE_FEATURE_END = SOURCE_EVIDENCE_FEATURE_START + SOURCE_EVIDENCE_FEATURE_DIM
COMMAND_SLOT_POSITION_FEATURE_START = SOURCE_EVIDENCE_FEATURE_END
COMMAND_SLOT_POSITION_FEATURE_END = (
    COMMAND_SLOT_POSITION_FEATURE_START + COMMAND_SLOT_POSITION_FEATURE_DIM
)
COMMAND_IDENTITY_FEATURE_START = COMMAND_SLOT_POSITION_FEATURE_END
COMMAND_IDENTITY_FEATURE_END = (
    COMMAND_IDENTITY_FEATURE_START + COMMAND_IDENTITY_CANDIDATE_FEATURE_DIM
)
CANDIDATE_FEATURE_DIM = (
    COMMAND_CANDIDATE_BASE_FEATURE_DIM
    + COMMAND_INTENT_COUNT
    + SOURCE_EVIDENCE_FEATURE_DIM
    + COMMAND_SLOT_POSITION_FEATURE_DIM
    + COMMAND_IDENTITY_CANDIDATE_FEATURE_DIM
)
COMMAND_CANDIDATE_FEATURE_GROUP_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "candidate_kind": ((0, 4),),
    "failure_signal": ((4, 7),),
    "lexical_overlap": ((7, 10),),
    "state_flags": ((10, 12),),
    "last_command": ((12, 15),),
    "intent": ((COMMAND_INTENT_FEATURE_START, COMMAND_INTENT_FEATURE_END),),
    "source_overlap": ((SOURCE_EVIDENCE_FEATURE_START, SOURCE_EVIDENCE_FEATURE_END),),
    "slot_position": (
        (COMMAND_SLOT_POSITION_FEATURE_START, COMMAND_SLOT_POSITION_FEATURE_END),
    ),
    "candidate_identity": ((COMMAND_IDENTITY_FEATURE_START, COMMAND_IDENTITY_FEATURE_END),),
}
COMMAND_CANDIDATE_FEATURE_GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "visible_shortcuts": (
        "lexical_overlap",
        "last_command",
        "intent",
        "source_overlap",
        "slot_position",
    ),
    "non_identity": (
        "candidate_kind",
        "failure_signal",
        "lexical_overlap",
        "state_flags",
        "last_command",
        "intent",
        "source_overlap",
        "slot_position",
    ),
    "all": tuple(COMMAND_CANDIDATE_FEATURE_GROUP_RANGES),
}
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_LAST_COMMAND_RE = re.compile(r"^last_command=(.*)$", re.MULTILINE)
_SOURCE_EVIDENCE_OMIT_LINE_RE = re.compile(
    r"^(?:last_command|last_command_intent|candidate_command_intents)=",
    re.MULTILINE,
)
_STRUCTURED_COMMAND_IDENTITY_RE = re.compile(
    r"\b(?:phase2j_)?command_identity_tokens\s*=\s*(.*?)(?=\s+(?:edit_scope|target_symbol|risk_label|command_candidate)=|[\n;]|$)",
    re.IGNORECASE,
)
_STRUCTURAL_PROBE_HASHES_JSON_RE = re.compile(
    r'("structural_probe_hashes"\s*:\s*)\[[^\]]*\]',
    re.IGNORECASE,
)
_IDENTITY_HASH_FIELD_JSON_RE = re.compile(
    r'("(?:expected_literal_hash|target_literal_hash|structural_probe_hash)"\s*:\s*)"[^"]*"',
    re.IGNORECASE,
)
_IDENTITY_FREE_TEXT_FIELD_RE = re.compile(
    r"\b(?:target_literal_hash|structural_probe_hash|target_line|target_col)\s*=\s*[^;\s]+",
    re.IGNORECASE,
)
_CANDIDATE_SECTION_RE = re.compile(
    r"\nCandidate commands:\n.*?(?=\nCandidate files:\n|\nHead constraints:\n|\Z)",
    re.DOTALL,
)
_CANDIDATE_VERIFICATION_COMMAND_SECTION_RE = re.compile(
    r"\nCandidate verification commands:\n.*?(?=\nCandidate files:\n|\nHead constraints:\n|\Z)",
    re.DOTALL,
)
_CANDIDATE_REPAIR_ACTION_SECTION_RE = re.compile(
    r"\nCandidate repair actions:\n.*?(?=\nCandidate commands:\n|\nCandidate files:\n|\nHead constraints:\n|\Z)",
    re.DOTALL,
)
_CANDIDATE_FILE_SECTION_RE = re.compile(
    r"\nCandidate files:\n.*?(?=\nHead constraints:\n|\Z)",
    re.DOTALL,
)
_PAIRWISE_VISIBLE_PREFIXES = (
    "failure_signal=",
    "debug_action_stage=",
    "source_inspected=",
    "last_command_intent=",
    "candidate_command_intents=",
    "goal_description=",
    "stdout_delta=",
    "stderr_delta=",
    "last_command=",
    "dirty_files=",
    "changed_paths=",
    "watched_paths=",
    "external_change_detected=",
    "stale_cache_detected=",
    "conflict_detected=",
    "command_candidate=",
    "risk_label=",
)
_RUNTIME_REPAIR_EVIDENCE_RE = re.compile(
    r"(?:Runtime-visible repair evidence|Prior runtime evidence):\s*"
    r"(\{.*?\})\s*(?:\n\n[A-Z][^\n]*:|\Z)",
    re.DOTALL,
)
_CANDIDATE_FIELD_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)=([^;\s]+)")


def normalize_command_candidate_feature_ablation_groups(
    groups: Iterable[str] | None,
) -> tuple[str, ...]:
    """Validate and expand candidate-feature ablation group names."""

    output: list[str] = []
    seen: set[str] = set()
    for raw_group in groups or ():
        for raw_name in str(raw_group).split(","):
            name = raw_name.strip()
            if not name:
                continue
            expanded = COMMAND_CANDIDATE_FEATURE_GROUP_ALIASES.get(name, (name,))
            for group_name in expanded:
                if group_name not in COMMAND_CANDIDATE_FEATURE_GROUP_RANGES:
                    supported = sorted(
                        set(COMMAND_CANDIDATE_FEATURE_GROUP_RANGES)
                        | set(COMMAND_CANDIDATE_FEATURE_GROUP_ALIASES)
                    )
                    raise ValueError(
                        f"unsupported command candidate feature ablation group "
                        f"{group_name!r}; supported={supported}"
                    )
                if group_name not in seen:
                    seen.add(group_name)
                    output.append(group_name)
    return tuple(output)


def zero_command_candidate_feature_groups(
    rows: list[list[float]],
    groups: Iterable[str] | None,
    *,
    feature_dim: int | None = None,
) -> list[list[float]]:
    """Zero selected candidate feature groups in-place and return rows."""

    normalized = normalize_command_candidate_feature_ablation_groups(groups)
    if not normalized:
        return rows
    ranges: list[tuple[int, int]] = []
    for group_name in normalized:
        ranges.extend(COMMAND_CANDIDATE_FEATURE_GROUP_RANGES[group_name])
    for row in rows:
        row_width = len(row) if feature_dim is None else min(len(row), feature_dim)
        for start, end in ranges:
            for index in range(start, min(end, row_width)):
                row[index] = 0.0
    return rows


def _tokens(text: str) -> set[str]:
    output: set[str] = set()
    for token in _TOKEN_RE.findall(text.lower()):
        if len(token) <= 1:
            continue
        output.add(token)
        for part in token.split("_"):
            if len(part) > 1:
                output.add(part)
    return output


def redact_structured_command_identity_text(text: str) -> str:
    """Hide Phase2J command-identity sidecar tokens from text baselines.

    The sidecar is a runtime receptor signal consumed by the NSI latent path.
    Keeping its raw tokens out of prompt/source-overlap features prevents the
    lexical baseline and lightweight candidate features from using the same
    evidence as the command-identity latent.
    """

    redacted = _STRUCTURED_COMMAND_IDENTITY_RE.sub("command_identity_tokens=<redacted>", text)
    redacted = _STRUCTURAL_PROBE_HASHES_JSON_RE.sub(r'\1["<redacted>"]', redacted)
    redacted = _IDENTITY_HASH_FIELD_JSON_RE.sub(r'\1"<redacted>"', redacted)
    redacted = _IDENTITY_FREE_TEXT_FIELD_RE.sub("<redacted_identity_field>", redacted)
    return redacted


def _visible_failure_signals(visible_text: str) -> dict[str, bool]:
    lowered = visible_text.lower()
    snapshot = "snapshot" in lowered and ("mismatch" in lowered or "update" in lowered)
    dependency = (
        "modulenotfounderror" in lowered
        or "no module named" in lowered
        or "missing dependency" in lowered
        or "dependency missing" in lowered
    )
    assertion = ("assertionerror" in lowered or "assertion failure" in lowered) and not snapshot
    return {"snapshot": snapshot, "dependency": dependency, "assertion": assertion}


def _without_candidate_sections(visible_state_text: str) -> str:
    """Remove candidate lists before computing state/candidate overlap.

    The native-head prompt includes candidate commands and richer candidate
    repair-action metadata for the backbone, but feature overlap must measure
    evidence outside candidate lists. Otherwise every candidate matches itself
    and the lightweight reranker learns a misleading shortcut instead of
    source/failure evidence.
    """

    text = _CANDIDATE_REPAIR_ACTION_SECTION_RE.sub("\n", visible_state_text)
    text = _CANDIDATE_VERIFICATION_COMMAND_SECTION_RE.sub("\n", text)
    text = _CANDIDATE_SECTION_RE.sub("\n", text)
    text = _CANDIDATE_FILE_SECTION_RE.sub("\n", text)
    return text


def _source_evidence_text(visible_state_text: str) -> str:
    """Return visible evidence that should identify the next command target.

    ``last_command`` is useful context for intent, but it is not reliable
    evidence that the next rerun should choose the same pytest target after
    source inspection has narrowed the failure. Candidate lists are also
    stripped before this function runs, so the overlap below cannot win by
    matching the candidate list itself.
    """

    lines = []
    redacted = redact_structured_command_identity_text(
        _without_candidate_sections(visible_state_text)
    )
    for raw_line in redacted.splitlines():
        line = raw_line.strip()
        if not line or _SOURCE_EVIDENCE_OMIT_LINE_RE.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def _discriminative_candidate_tokens(candidates: list[str]) -> list[set[str]]:
    token_rows = [_tokens(candidate) for candidate in candidates[:MAX_CANDIDATE_SLOTS]]
    if not token_rows:
        return []
    common_tokens = set.intersection(*token_rows) if all(token_rows) else set()
    return [set(tokens - common_tokens) for tokens in token_rows]


def command_candidate_source_overlap_rows(
    visible_state_text: str,
    candidates: list[str],
) -> list[list[float]]:
    """Score candidates against source/failure evidence only.

    This is a generic lexical baseline and feature source. It removes tokens
    common to all candidates, so shared command syntax such as ``python -m
    pytest`` cannot dominate same-intent command-slot decisions.
    """

    source_text = _source_evidence_text(visible_state_text)
    redacted_candidates = [redact_structured_command_identity_text(candidate) for candidate in candidates]
    candidate_tokens = _discriminative_candidate_tokens(redacted_candidates)
    visible_tokens = _tokens(source_text)
    visible_tokens = visible_tokens - set.intersection(*candidate_tokens) if candidate_tokens and all(candidate_tokens) else visible_tokens
    overlap_rows: list[tuple[float, float, float]] = []
    for index in range(MAX_CANDIDATE_SLOTS):
        command = candidates[index] if index < len(candidates) else ""
        tokens = candidate_tokens[index] if index < len(candidate_tokens) else set()
        if not command or not tokens or not visible_tokens:
            overlap_rows.append((0.0, 0.0, 0.0))
            continue
        overlap = tokens & visible_tokens
        overlap_count = min(len(overlap), 12) / 12.0
        candidate_overlap = len(overlap) / max(len(tokens), 1)
        visible_overlap = len(overlap) / max(len(visible_tokens), 1)
        overlap_rows.append((float(overlap_count), float(candidate_overlap), float(visible_overlap)))

    candidate_scores = [row[1] for row in overlap_rows]
    sorted_scores = sorted(candidate_scores, reverse=True)
    max_score = sorted_scores[0] if sorted_scores else 0.0
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    rows: list[list[float]] = []
    for overlap_count, candidate_overlap, visible_overlap in overlap_rows:
        is_best = candidate_overlap > 0.0 and candidate_overlap == max_score
        margin = candidate_overlap - (second_score if is_best else max_score)
        rows.append(
            [
                overlap_count,
                candidate_overlap,
                visible_overlap,
                1.0 if is_best else 0.0,
                float(margin),
            ]
        )
    return rows


def structured_command_identity_rows(
    visible_state_text: str,
    candidates: list[str],
) -> list[list[float]]:
    """Score candidates from structured runtime command-identity sidecar tokens.

    This path is intentionally separate from source-overlap evidence. The
    source-overlap baseline redacts the sidecar, while the NSI latent may read
    the raw runtime receptor state and compare those non-label tokens against
    candidate command identities.
    """

    identity_tokens: set[str] = set()
    for match in _STRUCTURED_COMMAND_IDENTITY_RE.finditer(visible_state_text):
        identity_tokens.update(_tokens(match.group(1)))
    candidate_tokens = _discriminative_candidate_tokens(candidates)
    rows: list[tuple[float, float, float]] = []
    for index in range(MAX_CANDIDATE_SLOTS):
        command = candidates[index] if index < len(candidates) else ""
        tokens = candidate_tokens[index] if index < len(candidate_tokens) else set()
        if not command or not tokens or not identity_tokens:
            rows.append((0.0, 0.0, 0.0))
            continue
        overlap = tokens & identity_tokens
        overlap_count = min(len(overlap), 12) / 12.0
        candidate_overlap = len(overlap) / max(len(tokens), 1)
        identity_overlap = len(overlap) / max(len(identity_tokens), 1)
        rows.append((float(overlap_count), float(candidate_overlap), float(identity_overlap)))

    candidate_scores = [row[1] for row in rows]
    sorted_scores = sorted(candidate_scores, reverse=True)
    max_score = sorted_scores[0] if sorted_scores else 0.0
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    output: list[list[float]] = []
    for overlap_count, candidate_overlap, identity_overlap in rows:
        is_best = candidate_overlap > 0.0 and candidate_overlap == max_score
        margin = candidate_overlap - (second_score if is_best else max_score)
        output.append(
            [
                overlap_count,
                candidate_overlap,
                identity_overlap,
                1.0 if is_best else 0.0,
                float(margin),
            ]
        )
    return output


def runtime_evidence_command_identity_rows(
    visible_state_text: str,
    candidates: list[str],
) -> list[list[float]]:
    """Score fixed candidates from label-free structured runtime evidence.

    Unlike source-overlap features, this receptor path may use structural
    evidence identifiers. They are produced by prior runtime observation and
    compared against fixed candidate metadata without reading a gold slot.
    """

    evidence = _runtime_repair_evidence(visible_state_text)
    return runtime_evidence_mapping_command_identity_rows(evidence, candidates)


def runtime_evidence_mapping_command_identity_rows(
    evidence: dict[str, Any],
    candidates: list[str],
) -> list[list[float]]:
    """Score fixed candidates directly from a structured receptor payload."""

    raw_tokens = _tokens_from_runtime_value(evidence)
    evidence_tokens: set[str] = set()
    for token in raw_tokens:
        normalized = token.strip().lower()
        if not normalized:
            continue
        evidence_tokens.add(normalized)
        if len(normalized) >= 12:
            evidence_tokens.add(normalized[:12])
        if len(normalized) >= 16:
            evidence_tokens.add(normalized[:16])
            evidence_tokens.add(f"structural_repair_{normalized[:12]}")
    scores: list[float] = []
    for index in range(MAX_CANDIDATE_SLOTS):
        candidate = candidates[index] if index < len(candidates) else ""
        lowered = candidate.lower()
        score = sum(1.0 for token in evidence_tokens if token and token in lowered)
        scores.append(float(score))
    best = max(scores) if scores else 0.0
    sorted_scores = sorted(scores, reverse=True)
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    unique_best = best > 0.0 and scores.count(best) == 1
    return [
        [
            score,
            score,
            score / max(len(evidence_tokens), 1),
            1.0 if unique_best and score == best else 0.0,
            float(score - second if unique_best and score == best else 0.0),
        ]
        for score in scores
    ]


def _normal_token(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\\", "/").strip().lower()


def _tokens_from_runtime_value(value: Any) -> set[str]:
    if isinstance(value, dict):
        tokens: set[str] = set()
        for nested in value.values():
            tokens.update(_tokens_from_runtime_value(nested))
        return tokens
    if isinstance(value, list):
        tokens = set()
        for nested in value:
            tokens.update(_tokens_from_runtime_value(nested))
        return tokens
    token = _normal_token(value)
    return {token} if token else set()


def _runtime_repair_evidence(visible_state_text: str) -> dict[str, Any]:
    match = _RUNTIME_REPAIR_EVIDENCE_RE.search(visible_state_text)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _runtime_verification_tokens(visible_state_text: str) -> set[str]:
    evidence = _runtime_repair_evidence(visible_state_text)
    if not evidence:
        return set()
    tokens: set[str] = set()
    tokens.update(_tokens_from_runtime_value(evidence.get("traceback_symbols")))
    tokens.update(_tokens_from_runtime_value(evidence.get("changed_files")))
    tokens.update(_tokens_from_runtime_value(evidence.get("watched_files")))
    tokens.update(_tokens_from_runtime_value(evidence.get("structural_probe_hashes")))
    literal = _normal_token(evidence.get("expected_literal_hash"))
    if literal:
        tokens.add(f"literal:{literal}")
    location = evidence.get("target_location")
    if isinstance(location, dict):
        path = _normal_token(location.get("path"))
        line = _normal_token(location.get("line"))
        col = _normal_token(location.get("col"))
        if path:
            tokens.add(path)
        if path and line and col:
            tokens.add(f"loc:{path}:{line}:{col}")
    return {token for token in tokens if token}


def _candidate_field_tokens(candidate_text: str) -> set[str]:
    fields = {key: value for key, value in _CANDIDATE_FIELD_RE.findall(candidate_text)}
    tokens: set[str] = set()
    for key in (
        "edit_scope",
        "target_symbol",
        "target_literal_hash",
        "structural_probe_hash",
        "verification_probe_token",
    ):
        value = _normal_token(fields.get(key))
        if not value:
            continue
        if key == "target_literal_hash":
            tokens.add(f"literal:{value}")
        else:
            tokens.add(value)
    path = _normal_token(fields.get("edit_scope"))
    line = _normal_token(fields.get("target_line"))
    col = _normal_token(fields.get("target_col"))
    if path and line and col:
        tokens.add(f"loc:{path}:{line}:{col}")
    return {token for token in tokens if token}


def runtime_verification_candidate_prediction(
    visible_state_text: str,
    candidates: list[str],
) -> int | None:
    """Predict candidate slot from runtime-visible repair evidence only.

    This guard is deliberately label-free: it compares evidence such as
    traceback symbols, target locations, watched files, and structural hashes
    against candidate metadata. It is used only as a consistency check for
    command-identity sidecars; it does not read gold slots or test IDs.
    """

    runtime_tokens = _runtime_verification_tokens(visible_state_text)
    if not runtime_tokens:
        return None
    scores: list[float] = []
    for index in range(MAX_CANDIDATE_SLOTS):
        candidate = candidates[index] if index < len(candidates) else ""
        candidate_tokens = _candidate_field_tokens(candidate)
        if not candidate or not candidate_tokens:
            scores.append(0.0)
            continue
        score = len(candidate_tokens & runtime_tokens)
        scores.append(float(score))
    best = max(scores) if scores else 0.0
    if best <= 0.0 or scores.count(best) != 1:
        return None
    return scores.index(best)


def identity_sidecar_consistent_with_runtime_evidence(
    visible_state_text: str,
    candidates: list[str],
    identity_scores: list[float],
) -> bool:
    """Return False when runtime-visible evidence contradicts identity sidecar.

    If runtime evidence cannot make a unique candidate distinction, the guard
    leaves sidecar features intact. This avoids silently disabling a valid
    sidecar on tasks that require non-lexical continuation or other receptors.
    """

    verification_slot = runtime_verification_candidate_prediction(
        visible_state_text,
        candidates,
    )
    if verification_slot is None:
        return True
    best_identity_score = max(identity_scores) if identity_scores else 0.0
    if best_identity_score <= 0.0 or identity_scores.count(best_identity_score) != 1:
        return True
    return identity_scores.index(best_identity_score) == verification_slot


def guard_command_identity_reference(
    visible_state_text: str,
    candidates: list[str],
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a copy with command-identity fields zeroed on visible contradiction."""

    guarded = dict(reference or {})
    identity_scores = [
        float(guarded.get(f"command_identity_slot:{index}", 0.0) or 0.0)
        for index in range(MAX_CANDIDATE_SLOTS)
    ]
    if identity_sidecar_consistent_with_runtime_evidence(
        visible_state_text,
        candidates,
        identity_scores,
    ):
        return guarded
    for index in range(MAX_CANDIDATE_SLOTS):
        guarded[f"command_identity_slot:{index}"] = 0.0
    guarded["command_identity_margin"] = 0.0
    guarded["command_identity_confidence"] = 0.0
    return guarded


def source_overlap_command_slot_prediction(
    visible_state_text: str,
    candidates: list[str],
) -> int:
    """Predict a command slot from visible source evidence without labels."""

    rows = command_candidate_source_overlap_rows(visible_state_text, candidates)
    if not rows:
        return 0
    return max(range(len(rows)), key=lambda index: (rows[index][1], rows[index][0], -index))


def compact_visible_state_for_candidate_pair(state_prompt: str) -> str:
    """Keep only receptor evidence needed for command candidate reranking.

    The full native-head state prompt is intentionally rich, but it can exceed
    the 256-token 7B canary budget before the pairwise candidate text is seen.
    Pairwise reranking needs a compact, non-oracle view of visible receptor
    evidence, not the full training instructions or candidate list.
    """

    text = redact_structured_command_identity_text(_without_candidate_sections(state_prompt))
    kept: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(_PAIRWISE_VISIBLE_PREFIXES):
            kept.append(line)
    return "\n".join(kept)


def build_candidate_pair_prompt(
    state_prompt: str,
    candidate_text: str,
    *,
    kind: str,
) -> str:
    """Build a truncation-safe pairwise candidate prompt.

    Candidate text is placed before compact receptor evidence so standard
    right-side tokenizer truncation cannot remove the very command being
    scored. The compact state is derived only from model-visible fields.
    """

    return "\n".join(
        [
            "Phase 2C candidate rerank input.",
            "Choose whether the candidate is the correct motor slot for the visible receptor state.",
            "",
            f"{kind} candidate under evaluation:",
            candidate_text,
            "",
            "Compact visible state evidence:",
            compact_visible_state_for_candidate_pair(state_prompt),
        ]
    )


def command_intent_for_text(text: str | None) -> str:
    command_lower = (text or "").lower()
    if (
        "pip install" in command_lower
        or "requirements" in command_lower
        or " install" in f" {command_lower}"
    ):
        return "dependency_install"
    if "snapshot" in command_lower and "update" in command_lower:
        return "snapshot_update"
    if "pytest" in command_lower:
        return "test_rerun"
    return "other"


def command_intent_index(text: str | None) -> int:
    return COMMAND_INTENT_ORDER.index(command_intent_for_text(text))


def command_candidate_intent_indices(candidates: list[str]) -> list[int]:
    rows: list[int] = []
    for index in range(MAX_CANDIDATE_SLOTS):
        command = candidates[index] if index < len(candidates) else ""
        rows.append(command_intent_index(command))
    return rows


def pairwise_command_candidate_mask(
    candidates: list[str],
    policy: str = "all",
    *,
    visible_state_text: str | None = None,
    top_k: int | None = None,
) -> list[bool]:
    """Return candidate positions that should receive pairwise cross-encoding.

    ``ambiguous_intent`` is intentionally label-free: it only enables pairwise
    scoring for candidate commands that compete with another candidate in the
    same coarse command-intent bucket.
    """

    if policy not in PAIRWISE_COMMAND_POLICIES:
        raise ValueError(f"unsupported pairwise_command_policy={policy!r}")
    bounded = list(candidates)[:MAX_CANDIDATE_SLOTS]
    intent_counts: dict[str, int] = {}
    intents = [command_intent_for_text(command) for command in bounded]
    for intent in intents:
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    if policy == "all":
        rows = [index < len(bounded) for index in range(MAX_CANDIDATE_SLOTS)]
    else:
        rows = [False] * MAX_CANDIDATE_SLOTS
        for index, intent in enumerate(intents):
            rows[index] = intent_counts.get(intent, 0) > 1
    if top_k is None or top_k <= 0 or visible_state_text is None:
        return rows
    overlap_rows = command_candidate_source_overlap_rows(visible_state_text, bounded)
    selected = [False] * MAX_CANDIDATE_SLOTS
    grouped: dict[str, list[int]] = {}
    for index, enabled in enumerate(rows):
        if not enabled or index >= len(intents):
            continue
        grouped.setdefault(intents[index], []).append(index)
    for group in grouped.values():
        ranked = sorted(
            group,
            key=lambda index: (
                overlap_rows[index][1],
                overlap_rows[index][0],
                overlap_rows[index][2],
                overlap_rows[index][4],
                -index,
            ),
            reverse=True,
        )
        for index in ranked[:top_k]:
            selected[index] = True
    return selected


def command_intent_target(command: str | None) -> int:
    if not command:
        return -100
    return command_intent_index(command)


def command_candidate_feature_rows(
    visible_state_text: str,
    candidates: list[str],
    *,
    nsi_reference: dict[str, Any] | None = None,
) -> list[list[float]]:
    """Build lightweight visible-state/candidate features.

    These features are not oracle labels. They expose generic lexical and
    affordance signals to the native command head so 7B does not need an
    expensive full cross-encoder pass for every command candidate.
    """

    evidence_text = redact_structured_command_identity_text(
        _without_candidate_sections(visible_state_text)
    )
    visible_tokens = _tokens(evidence_text)
    signals = _visible_failure_signals(evidence_text)
    visible_lower = evidence_text.lower()
    source_inspected = (
        "source inspected" in visible_lower
        or "rerun the targeted failing test" in visible_lower
    )
    rerun_visible = "rerun" in visible_lower or "targeted failing test" in visible_lower
    last_command_match = _LAST_COMMAND_RE.search(evidence_text)
    last_command = (last_command_match.group(1).strip() if last_command_match else "")
    last_command_lower = last_command.lower()
    last_command_is_test_rerun = (
        "pytest" in last_command_lower and "--snapshot-update" not in last_command_lower
    )
    nsi_reference = guard_command_identity_reference(
        visible_state_text,
        candidates,
        nsi_reference,
    )
    identity_scores = [
        float((nsi_reference or {}).get(f"command_identity_slot:{index}", 0.0) or 0.0)
        for index in range(MAX_CANDIDATE_SLOTS)
    ]
    best_identity_score = max(identity_scores) if identity_scores else 0.0
    best_identity_slot = (
        identity_scores.index(best_identity_score)
        if best_identity_score > 0.0 and identity_scores.count(best_identity_score) == 1
        else -1
    )
    identity_margin = float((nsi_reference or {}).get("command_identity_margin", 0.0) or 0.0)
    identity_confidence = float(
        (nsi_reference or {}).get("command_identity_confidence", 0.0) or 0.0
    )
    rows: list[list[float]] = []
    source_overlap_rows = command_candidate_source_overlap_rows(visible_state_text, candidates)
    for index in range(MAX_CANDIDATE_SLOTS):
        command = candidates[index] if index < len(candidates) else ""
        command_lower = command.lower()
        command_tokens = _tokens(command)
        overlap = command_tokens & visible_tokens
        overlap_count = min(len(overlap), 12) / 12.0
        candidate_overlap = len(overlap) / max(len(command_tokens), 1) if command_tokens else 0.0
        visible_overlap = len(overlap) / max(len(visible_tokens), 1) if visible_tokens else 0.0
        is_pytest = "pytest" in command_lower
        updates_snapshot = "snapshot" in command_lower and "update" in command_lower
        installs_dependency = (
            "pip install" in command_lower
            or "requirements" in command_lower
            or " install" in f" {command_lower}"
        )
        matches_last_command = bool(command and command_lower == last_command_lower)
        intent = command_intent_for_text(command)
        intent_one_hot = [
            1.0 if intent == intent_name and command else 0.0
            for intent_name in COMMAND_INTENT_ORDER
        ]
        slot_position = [
            1.0 if command and index == slot_index else 0.0
            for slot_index in range(MAX_CANDIDATE_SLOTS)
        ]
        identity_is_best = command and index == best_identity_slot
        identity_features = [
            identity_scores[index] if command else 0.0,
            1.0 if identity_is_best else 0.0,
            identity_margin if identity_is_best else 0.0,
            identity_confidence if identity_is_best else 0.0,
        ]
        rows.append(
            [
                1.0 if command else 0.0,
                1.0 if is_pytest else 0.0,
                1.0 if updates_snapshot else 0.0,
                1.0 if installs_dependency else 0.0,
                1.0 if signals["snapshot"] and updates_snapshot else 0.0,
                1.0 if signals["dependency"] and installs_dependency else 0.0,
                1.0 if signals["assertion"] and is_pytest and not updates_snapshot else 0.0,
                float(overlap_count),
                float(candidate_overlap),
                float(visible_overlap),
                1.0 if source_inspected else 0.0,
                1.0 if rerun_visible else 0.0,
                1.0 if matches_last_command else 0.0,
                1.0 if source_inspected and matches_last_command else 0.0,
                1.0
                if source_inspected and last_command_is_test_rerun and matches_last_command
                else 0.0,
                *intent_one_hot,
                *source_overlap_rows[index],
                *slot_position,
                *identity_features,
            ]
        )
    return rows
