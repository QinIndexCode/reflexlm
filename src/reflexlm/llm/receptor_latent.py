from __future__ import annotations

from reflexlm.llm.candidate_features import (
    command_candidate_source_overlap_rows,
    runtime_evidence_command_identity_rows,
    runtime_evidence_mapping_command_identity_rows,
    structured_command_identity_rows,
)
from reflexlm.models.features import MAX_CANDIDATE_SLOTS, candidate_commands
from reflexlm.schema import SystemStateFrame


RECEPTOR_FAILURE_SIGNAL_ORDER = (
    "other",
    "snapshot_update",
    "dependency_install",
    "assertion_inspection",
    "source_inspected",
)
DEBUG_ACTION_STAGE_ORDER = (
    "other",
    "raw_failure_output",
    "parsed_failure_summary",
    "source_inspected",
)
DEBUG_ACTION_STAGE_LATENT_FIELDS = tuple(
    f"debug_action_stage:{stage}" for stage in DEBUG_ACTION_STAGE_ORDER
)
DESCRIPTOR_FAILURE_FAMILY_ORDER = (
    "other",
    "unknown_runtime_failure",
    "attribute_missing_runtime",
    "missing_import_or_symbol_runtime",
    "syntax_load_failure_runtime",
    "assertion_behavior_mismatch_runtime",
)
DESCRIPTOR_FAILURE_FAMILY_LATENT_FIELDS = tuple(
    f"descriptor_failure_family:{family}"
    for family in DESCRIPTOR_FAILURE_FAMILY_ORDER
)
COMMAND_IDENTITY_SLOT_FIELDS = tuple(
    f"command_identity_slot:{index}" for index in range(MAX_CANDIDATE_SLOTS)
)
COMMAND_IDENTITY_SUMMARY_FIELDS = (
    "command_identity_margin",
    "command_identity_confidence",
)
COMMAND_IDENTITY_LATENT_FIELDS = (
    *COMMAND_IDENTITY_SLOT_FIELDS,
    *COMMAND_IDENTITY_SUMMARY_FIELDS,
)


def receptor_failure_signal(state: SystemStateFrame) -> str:
    """Low-level terminal receptor feature derived from raw stdout/stderr.

    This deliberately uses only model-observable system state fields. It is not
    a scenario label, recovery hint, oracle action, or test answer. Phase2F can
    compress the cortex text prompt while still injecting this sensory feature
    through the native NSI latent path.
    """

    visible = f"{state.terminal.stderr_delta} {state.terminal.stdout_delta}".lower()
    if "source inspected" in visible or "rerun the targeted failing test" in visible:
        return "source_inspected"
    if "snapshot" in visible and ("mismatch" in visible or "update" in visible):
        return "snapshot_update"
    if (
        "modulenotfounderror" in visible
        or "no module named" in visible
        or "missing dependency" in visible
        or "dependency missing" in visible
    ):
        return "dependency_install"
    if "assertionerror" in visible or "assertion failure" in visible:
        return "assertion_inspection"
    return "other"


def debug_action_stage_signal(state: SystemStateFrame) -> str:
    """Visible debug-transition stage, independent of oracle action labels.

    The signal separates a raw terminal-error frame from a parsed failure frame
    and a post-source-inspection frame. It uses only runtime-visible state:
    stdout/stderr deltas, dirty files, and watched paths. It deliberately does
    not read recovery hints, scenario names, target action labels, or sealed
    failure metadata.
    """

    stdout = state.terminal.stdout_delta.strip()
    stderr = state.terminal.stderr_delta.strip()
    visible = f"{stdout} {stderr}".lower()
    if "source inspected" in visible or "rerun the targeted failing test" in visible:
        return "source_inspected"
    if stderr:
        return "raw_failure_output"
    if stdout.lower().startswith("parsed failure:"):
        return "parsed_failure_summary"
    if state.filesystem.dirty_files and any(path in state.filesystem.watched_paths for path in state.filesystem.dirty_files):
        return "parsed_failure_summary"
    return "other"


def descriptor_failure_family_signal(state: SystemStateFrame) -> str:
    """Runtime-visible descriptor repair family, independent of oracle labels."""

    visible = f"{state.terminal.stderr_delta} {state.terminal.stdout_delta}".lower()
    if "attributeerror" in visible or "has no attribute" in visible:
        return "attribute_missing_runtime"
    if (
        "nameerror" in visible
        or "importerror" in visible
        or "modulenotfounderror" in visible
        or "is not defined" in visible
        or "no module named" in visible
    ):
        return "missing_import_or_symbol_runtime"
    if (
        "syntaxerror" in visible
        or "indentationerror" in visible
        or "unexpected indent" in visible
        or "invalid syntax" in visible
    ):
        return "syntax_load_failure_runtime"
    if "assertionerror" in visible:
        return "assertion_behavior_mismatch_runtime"
    return "other"


def _command_identity_visible_text(state: SystemStateFrame) -> str:
    """Observable receptor evidence for command identity without candidate text.

    The command allowlist is passed separately to the scorer below. Keeping it
    out of this text prevents the latent from winning by self-overlap with the
    candidate list while still allowing source/error evidence to identify a
    command slot.
    """

    return "\n".join(
        [
            f"description={state.goal.description}",
            f"stdout_delta={state.terminal.stdout_delta}",
            f"stderr_delta={state.terminal.stderr_delta}",
            f"last_command={state.terminal.last_command}",
            f"dirty_files={','.join(state.filesystem.dirty_files)}",
            f"changed_paths={','.join(state.filesystem.changed_paths)}",
            f"watched_paths={','.join(state.filesystem.watched_paths)}",
            f"external_change_detected={state.filesystem.external_change_detected}",
            f"stale_cache_detected={state.filesystem.stale_cache_detected}",
            f"conflict_detected={state.filesystem.conflict_detected}",
        ]
    )


def runtime_command_identity_signal(state: SystemStateFrame) -> dict[str, float]:
    """Label-free command identity latent derived from visible receptor state.

    This deliberately does not inspect the oracle action, gold command slot,
    scenario profile, hidden recovery hint, or sealed-failure metadata. It only
    measures how strongly observable source/error evidence distinguishes each
    allowed command candidate.
    """

    candidates = candidate_commands(state)
    visible_text = _command_identity_visible_text(state)
    receptor_evidence = state.runtime_evidence.model_dump(mode="json", exclude_none=True)
    rows = runtime_evidence_mapping_command_identity_rows(receptor_evidence, candidates)
    if not any(float(row[1]) > 0.0 for row in rows):
        rows = structured_command_identity_rows(visible_text, candidates)
    if not any(float(row[1]) > 0.0 for row in rows):
        rows = runtime_evidence_command_identity_rows(visible_text, candidates)
    if not any(float(row[1]) > 0.0 for row in rows):
        rows = command_candidate_source_overlap_rows(
            visible_text,
            candidates,
        )
    slot_scores = [
        float(rows[index][1]) if index < len(rows) else 0.0
        for index in range(MAX_CANDIDATE_SLOTS)
    ]
    sorted_scores = sorted(slot_scores, reverse=True)
    best = sorted_scores[0] if sorted_scores else 0.0
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    unique_best = best > 0.0 and slot_scores.count(best) == 1
    payload = {
        field: slot_scores[index]
        for index, field in enumerate(COMMAND_IDENTITY_SLOT_FIELDS)
    }
    payload["command_identity_margin"] = float(best - second if unique_best else 0.0)
    payload["command_identity_confidence"] = float(best if unique_best else 0.0)
    return payload
