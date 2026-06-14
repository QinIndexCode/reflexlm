from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2m_external_generalization import (
    BASELINE_METHODS,
    compute_phase2m_baseline_predictions,
)


EVIDENCE_DENSITIES = ("low", "medium", "high")
CANDIDATE_COUNTS = (2, 3, 4)
CONTINUATION_DEPTHS = ("one_step", "two_step", "stale_state_refresh")
AMBIGUITY_CLASSES = ("same_intent_command", "same_file_read", "stage_transition")
TRACE_TYPES = (
    "test_failure_traceback_to_symbol",
    "changed_file_to_watched_test",
    "module_ownership_to_command",
    "stale_state_refresh",
)
NATURAL_TEST_TARGETS = (
    ("billing", "rounding", "invoice_total", "tax_boundary", "discount_stack", "currency_edge"),
    ("search", "ranking", "query_parser", "facets", "pagination", "synonym_map"),
    ("auth", "session", "refresh_flow", "permission_matrix", "csrf_guard", "token_rotation"),
    ("orders", "fulfillment", "stock_release", "carrier_retry", "address_rules", "refund_path"),
    ("reports", "exports", "csv_stream", "date_bucket", "timezone_cutoff", "summary_cache"),
    ("api", "contracts", "error_shape", "pagination_link", "idempotency", "rate_window"),
)


def _tokens(text: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if token}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _baseline_by_overlap(text: str, commands: list[str]) -> str:
    text_tokens = _tokens(text)
    scored = [
        (len(text_tokens & _tokens(command)), -index, command)
        for index, command in enumerate(commands)
    ]
    return max(scored)[2]


def _wrong_baseline(expected: str, commands: list[str]) -> str:
    for command in commands:
        if command != expected:
            return command
    return expected


def _trace_type_suffix(trace_type: str) -> str:
    return {
        "test_failure_traceback_to_symbol": "traceback_symbol",
        "changed_file_to_watched_test": "watched_file",
        "module_ownership_to_command": "module_owner",
        "stale_state_refresh": "stale_refresh",
    }[trace_type]


def build_phase2m_synthetic_safe_rows(
    *,
    split: str,
    count: int,
    repo_prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(count):
        evidence_density = EVIDENCE_DENSITIES[index % len(EVIDENCE_DENSITIES)]
        candidate_count = CANDIDATE_COUNTS[index % len(CANDIDATE_COUNTS)]
        continuation_depth = CONTINUATION_DEPTHS[index % len(CONTINUATION_DEPTHS)]
        ambiguity_class = AMBIGUITY_CLASSES[index % len(AMBIGUITY_CLASSES)]
        trace_type = TRACE_TYPES[index % len(TRACE_TYPES)]
        repo_id = f"{repo_prefix}_repo_{index % 5}"
        module = f"module_{index:03d}"
        symbol = f"{module}.handle_case"
        expected_slot = index % candidate_count
        suffix = _trace_type_suffix(trace_type)
        commands = [
            (
                "python -m pytest -q "
                f"tests/{repo_id}/test_{module}_candidate_{slot}.py::test_{suffix}"
            )
            for slot in range(candidate_count)
        ]
        expected_command = commands[expected_slot]
        watched_files = [
            f"tests/{repo_id}/test_{module}_candidate_{expected_slot}.py",
        ]
        if evidence_density in {"medium", "high"}:
            watched_files.append(f"tests/{repo_id}/test_{module}_support.py")
        if evidence_density == "high":
            watched_files.append(f"tests/{repo_id}/test_{module}_integration.py")
        runtime_visible_evidence = {
            "traceback_symbols": [symbol],
            "changed_files": [f"src/{repo_id}/{module}.py"],
            "watched_files": watched_files,
            "module_owner": f"{repo_id}.{module}.owner",
            "prior_read_summary": (
                f"Prior read linked {symbol} to {watched_files[0]} "
                f"under {continuation_depth} continuation."
            ),
            "stale_state_refresh": continuation_depth == "stale_state_refresh",
        }
        current_visible_text = (
            "External repository test failure. Same-intent candidate commands are available; "
            "select the bounded rerun command using runtime-visible traceback, changed-file, "
            "watched-file, module-owner, and continuation evidence. "
            f"ambiguity={ambiguity_class}; density={evidence_density}."
        )
        source_overlap_prediction = _baseline_by_overlap(current_visible_text, commands)
        native_head_only_prediction = _wrong_baseline(expected_command, commands)
        row = {
            "trace_id": f"{split}:{repo_id}:{index}",
            "split": split,
            "source_kind": "synthetic_safe_repo",
            "repo_id": repo_id,
            "repo_url_or_origin": f"synthetic://phase2m/{repo_id}",
            "commit_hash": _sha256(f"{repo_id}:{index}")[:40],
            "license_or_synthetic_origin": "synthetic-safe phase2m external-generalization fixture",
            "current_visible_text": current_visible_text,
            "runtime_visible_evidence": runtime_visible_evidence,
            "command_candidates": [
                {"command": command, "intent": "test_rerun"}
                for command in commands
            ],
            "expected_command": expected_command,
            "baselines": {
                "source_overlap": source_overlap_prediction,
                "native_head_only": native_head_only_prediction,
                "continuation_only": native_head_only_prediction,
                "prompt_only": source_overlap_prediction,
                "react": source_overlap_prediction,
            },
            "difficulty": {
                "evidence_density": evidence_density,
                "candidate_count": candidate_count,
                "continuation_depth": continuation_depth,
                "ambiguity_class": ambiguity_class,
                "trace_type": trace_type,
            },
        }
        row["trace_hash"] = _sha256(json.dumps(row, ensure_ascii=False, sort_keys=True))
        rows.append(row)
    return rows


def _measured_baseline_payload(row: dict[str, Any]) -> tuple[dict[str, str | None], dict[str, dict[str, Any]]]:
    predictions = compute_phase2m_baseline_predictions(row)
    metadata = {
        name: {
            "measured": True,
            "method": method,
            "uses_expected_command": False,
            "uses_sealed_feedback": False,
        }
        for name, method in BASELINE_METHODS.items()
    }
    return predictions, metadata


def build_phase2m_v2_synthetic_safe_rows(
    *,
    split: str,
    count: int,
    repo_prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(count):
        evidence_density = EVIDENCE_DENSITIES[index % len(EVIDENCE_DENSITIES)]
        candidate_count = CANDIDATE_COUNTS[index % len(CANDIDATE_COUNTS)]
        continuation_depth = CONTINUATION_DEPTHS[index % len(CONTINUATION_DEPTHS)]
        ambiguity_class = AMBIGUITY_CLASSES[index % len(AMBIGUITY_CLASSES)]
        trace_type = TRACE_TYPES[index % len(TRACE_TYPES)]
        domain, component, *behaviors = NATURAL_TEST_TARGETS[index % len(NATURAL_TEST_TARGETS)]
        repo_id = f"{repo_prefix}_{domain}_{index % 7}"
        module = f"{domain}.{component}"
        expected_slot = index % candidate_count
        selected_behaviors = behaviors[:candidate_count]
        commands = [
            (
                "python -m pytest -q "
                f"tests/{domain}/test_{component}_{behavior}.py::test_{behavior}_regression"
            )
            for behavior in selected_behaviors
        ]
        expected_command = commands[expected_slot]
        expected_behavior = selected_behaviors[expected_slot]
        sibling_behavior = selected_behaviors[(expected_slot + 1) % candidate_count]
        watch_keys = [
            _sha256(f"{repo_id}:{domain}:{component}:{behavior}")[:12]
            for behavior in selected_behaviors
        ]
        active_watch_key = watch_keys[expected_slot]
        watched_files = [
            f"tests/{domain}/test_{component}_suite.py",
        ]
        if evidence_density in {"medium", "high"}:
            watched_files.append(f"tests/{domain}/test_{component}_shared.py")
        if evidence_density == "high":
            watched_files.append(f"tests/{domain}/test_{component}_integration.py")
        runtime_visible_evidence = {
            "traceback_symbols": [f"{module}.dispatch_failure"],
            "changed_files": [f"src/{domain}/{component}.py"],
            "watched_files": watched_files,
            "module_owner": f"{domain}.{component}",
            "active_watch_key": active_watch_key,
            "prior_read_summary": (
                f"Prior read connected {module}.dispatch_failure to watch key "
                f"{active_watch_key} after reviewing adjacent behavior group."
            ),
            "stale_state_refresh": continuation_depth == "stale_state_refresh",
        }
        current_visible_text = (
            "External repository regression triage. Same-intent pytest rerun commands "
            "are available, but the current message intentionally gives only module-level "
            "context; use runtime-visible traceback, watched-file, ownership, and "
            f"continuation evidence. module={domain}.{component}; "
            f"ambiguity={ambiguity_class}; density={evidence_density}."
        )
        row = {
            "trace_id": f"{split}:{repo_id}:{index}",
            "split": split,
            "source_kind": "synthetic_safe_repo",
            "repo_id": repo_id,
            "repo_url_or_origin": f"synthetic://phase2m/v2/{repo_id}",
            "commit_hash": _sha256(f"phase2m-v2:{repo_id}:{index}")[:40],
            "license_or_synthetic_origin": "synthetic-safe phase2m v2 strict plumbing fixture",
            "current_visible_text": current_visible_text,
            "runtime_visible_evidence": runtime_visible_evidence,
            "command_candidates": [
                {
                    "command": command,
                    "intent": "test_rerun",
                    "module_owner": f"{domain}.{component}",
                    "watch_key": watch_keys[slot],
                    "test_group": selected_behaviors[slot],
                }
                for slot, command in enumerate(commands)
            ],
            "expected_command": expected_command,
            "difficulty": {
                "evidence_density": evidence_density,
                "candidate_count": candidate_count,
                "continuation_depth": continuation_depth,
                "ambiguity_class": ambiguity_class,
                "trace_type": trace_type,
            },
        }
        baselines, metadata = _measured_baseline_payload(row)
        row["baselines"] = baselines
        row["baseline_metadata"] = metadata
        row["trace_hash"] = _sha256(json.dumps(row, ensure_ascii=False, sort_keys=True))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate non-sealed synthetic-safe Phase2M raw traces for local smoke training."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-count", type=int, default=48)
    parser.add_argument("--val-count", type=int, default=24)
    parser.add_argument("--holdout-count", type=int, default=12)
    parser.add_argument("--variant", choices=["v1", "v2_strict"], default="v1")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    builder = (
        build_phase2m_v2_synthetic_safe_rows
        if args.variant == "v2_strict"
        else build_phase2m_synthetic_safe_rows
    )
    splits = {
        "train": builder(
            split="train", count=args.train_count, repo_prefix="train"
        ),
        "val": builder(
            split="val", count=args.val_count, repo_prefix="val"
        ),
        "holdout": builder(
            split="holdout", count=args.holdout_count, repo_prefix="holdout"
        ),
    }
    for split, rows in splits.items():
        _write_jsonl(output_root / f"{split}.raw.jsonl", rows)
    manifest = {
        "generator": "phase2m_synthetic_safe_trace_generator",
        "variant": args.variant,
        "sealed_v3_used": False,
        "splits": {
            split: {"path": str(output_root / f"{split}.raw.jsonl"), "rows": len(rows)}
            for split, rows in splits.items()
        },
        "dimensions": {
            "evidence_densities": list(EVIDENCE_DENSITIES),
            "candidate_counts": list(CANDIDATE_COUNTS),
            "continuation_depths": list(CONTINUATION_DEPTHS),
            "ambiguity_classes": list(AMBIGUITY_CLASSES),
            "trace_types": list(TRACE_TYPES),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
