from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


BENCHMARK_FAMILY = "phase2ax_package_loaded_counterfactual_repair"
CLAIM_BOUNDARY = (
    "phase2ax_nonsealed_package_loaded_counterfactual_repair_before_training"
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    if not file.exists():
        return []
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _expected_action(row: dict[str, Any]) -> str:
    return str(row.get("expected_repair_action") or "")


def _repo_key(row: dict[str, Any]) -> str:
    return f"{row.get('repo_origin')}@{row.get('repo_commit')}"


def _prior_evidence(row: dict[str, Any]) -> dict[str, Any]:
    evidence = row.get("runtime_visible_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    target = row.get("learned_patch_descriptor_target")
    target = target if isinstance(target, dict) else {}
    return {
        "version": "phase2ax_prior_runtime_evidence_v1",
        "source_task_id": row.get("task_id"),
        "changed_files": evidence.get("changed_files") or [],
        "watched_files": evidence.get("watched_files") or [],
        "structural_probe_hashes": evidence.get("structural_probe_hashes") or [],
        "repair_modes": evidence.get("repair_modes") or [],
        "descriptor_operation": target.get("operation"),
        "descriptor_template": target.get("after_fragment_template_id"),
        "target_path_hash": _sha256(target.get("target_path") or ""),
        "target_symbol_hash": (
            target.get("literal_or_symbol_payload", {}).get("target_symbol_hash")
            if isinstance(target.get("literal_or_symbol_payload"), dict)
            else None
        ),
    }


def _masked_current_surface(pair_index: int, candidate_count: int) -> dict[str, Any]:
    return {
        "version": "phase2ax_masked_current_repair_surface_v1",
        "same_intent_candidates": True,
        "candidate_count": candidate_count,
        "runtime_visible_current_text": (
            "The current repair surface exposes the same bounded patch candidates "
            "and verification contract for both counterfactual members. The "
            "decisive evidence must come from prior runtime context."
        ),
        "shared_surface_id": f"phase2ax_pair_{pair_index:05d}",
        "hidden_gold_or_slot_marker": False,
    }


def _counterfactual_candidates(row_a: dict[str, Any], row_b: dict[str, Any]) -> list[dict[str, Any]]:
    actions = [_expected_action(row_a), _expected_action(row_b)]
    candidates: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        candidates.append(
            {
                "repair_action": action,
                "intent": "apply_patch_and_rerun_tests",
                "edit_scope": "bounded_public_source_patch",
                "description": (
                    "Counterfactual bounded repair candidate with identical current "
                    "surface text; select using prior runtime evidence only."
                ),
                "structural_probe_hash": f"phase2ax_masked_probe_{index}",
                "target_symbol": "phase2ax_masked_symbol",
            }
        )
    return candidates


def _convert_member(
    source: dict[str, Any],
    peer: dict[str, Any],
    *,
    pair_index: int,
    member: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = _counterfactual_candidates(source, peer) if member == "a" else _counterfactual_candidates(peer, source)
    if member == "b":
        # Preserve identical candidate order across pair members.
        candidates = _counterfactual_candidates(peer, source)
        expected = _expected_action(source)
    else:
        expected = _expected_action(source)
    surface = _masked_current_surface(pair_index, len(candidates))
    prior = _prior_evidence(source)
    row = dict(source)
    row.update(
        {
            "task_id": f"phase2ax:pair_{pair_index:05d}:{member}",
            "benchmark_family": BENCHMARK_FAMILY,
            "claim_boundary": CLAIM_BOUNDARY,
            "phase2ax_counterfactual_repair": True,
            "phase2ax_pair_id": f"phase2ax_pair_{pair_index:05d}",
            "phase2ax_pair_member": member,
            "current_visible_text": surface["runtime_visible_current_text"],
            "phase2ax_current_repair_surface": surface,
            "phase2ax_prior_runtime_evidence": prior,
            "runtime_visible_evidence": {
                "phase2ax_masked_current_surface": surface,
                "source_repo_observed_read_only": True,
                "execution_sandbox_used": True,
            },
            "repair_candidates": candidates,
            "candidate_policy_commands": [
                (
                    "phase2ax_apply_counterfactual_repair_candidate "
                    f"--repair-action {candidate['repair_action']} "
                    "--verify <masked_current_verification_contract>"
                )
                for candidate in candidates
            ],
            "expected_repair_action": expected,
            "runtime_visible_contract": {
                "no_candidate_slot_marker": True,
                "no_gold_hint": True,
                "no_sealed_feedback": True,
                "public_repo_origin": True,
                "no_freeform_patch_generation": True,
                "counterfactual_prior_required": True,
                "current_surface_identical_with_pair_member": True,
            },
            "sealed_feedback_used": False,
            "difficulty_axes": sorted(
                set(source.get("difficulty_axes") or [])
                | {
                    "package_loaded_counterfactual_repair",
                    "same_current_surface",
                    "prior_runtime_evidence_required",
                }
            ),
        }
    )
    metadata = {
        "task_id": row["task_id"],
        "pair_id": row["phase2ax_pair_id"],
        "member": member,
        "repo_key": _repo_key(source),
        "source_task_id": source.get("task_id"),
        "peer_task_id": peer.get("task_id"),
        "current_surface_hash": _sha256(surface),
        "prior_context_hash": _sha256(prior),
        "candidate_actions_hash": _sha256([candidate["repair_action"] for candidate in candidates]),
        "expected_repair_action": expected,
        "expected_slot": [candidate["repair_action"] for candidate in candidates].index(expected),
        "artifact_paths": row.get("artifact_paths"),
    }
    row["phase2ax_metadata_hash"] = _sha256(metadata)
    return row, metadata


def build_phase2ax_package_loaded_counterfactual_repair(
    *,
    source_tasks_jsonl: str | Path,
    output_jsonl: str | Path,
    metadata_json: str | Path,
    max_pairs: int = 32,
    min_pairs: int = 16,
) -> dict[str, Any]:
    source_rows = [
        row
        for row in _read_jsonl(source_tasks_jsonl)
        if row.get("source_kind") == "public_repo"
        and row.get("sealed_feedback_used") is not True
        and _expected_action(row)
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        groups[_repo_key(row)].append(row)

    output_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    pair_index = 0
    for repo_key in sorted(groups):
        rows = groups[repo_key]
        for left, right in zip(rows[0::2], rows[1::2]):
            if pair_index >= max_pairs:
                break
            if _expected_action(left) == _expected_action(right):
                continue
            row_a, meta_a = _convert_member(left, right, pair_index=pair_index, member="a")
            row_b, meta_b = _convert_member(right, left, pair_index=pair_index, member="b")
            output_rows.extend([row_a, row_b])
            metadata_rows.extend([meta_a, meta_b])
            pair_index += 1
        if pair_index >= max_pairs:
            break

    report = {
        "artifact_family": "phase2ax_package_loaded_counterfactual_repair_builder",
        "passed": pair_index >= min_pairs,
        "benchmark_family": BENCHMARK_FAMILY,
        "claim_boundary": CLAIM_BOUNDARY,
        "pair_count": pair_index,
        "row_count": len(output_rows),
        "min_pairs": min_pairs,
        "max_pairs": max_pairs,
        "source_rows": len(source_rows),
        "output_jsonl": str(Path(output_jsonl)),
        "metadata_json": str(Path(metadata_json)),
        "blocked_actions": [] if pair_index >= min_pairs else ["do_not_train_phase2ax_until_min_pairs_exist"],
        "notes": [
            "Rows preserve public repo patch/test artifacts but mask current direct identity cues.",
            "The decisive variable is phase2ax_prior_runtime_evidence, not a gold slot marker.",
            "This builder does not use sealed v3 content or sealed failure feedback.",
        ],
    }
    _write_jsonl(output_jsonl, output_rows)
    _write_json(metadata_json, metadata_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AX package-loaded counterfactual repair tasks."
    )
    parser.add_argument("--source-tasks-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--min-pairs", type=int, default=16)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2ax_package_loaded_counterfactual_repair(
        source_tasks_jsonl=args.source_tasks_jsonl,
        output_jsonl=args.output_jsonl,
        metadata_json=args.metadata_json,
        max_pairs=args.max_pairs,
        min_pairs=args.min_pairs,
    )
    _write_json(args.output_report_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
