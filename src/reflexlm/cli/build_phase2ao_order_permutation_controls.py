from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


IDENTITY_PREFIX = "command_identity_slot:"


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    output.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _candidate_count(row: dict[str, Any]) -> int:
    candidates = row.get("candidate_commands")
    return len(candidates) if isinstance(candidates, list) else 0


def _scores(row: dict[str, Any]) -> list[float]:
    nsi = row.get("nsi_reference") if isinstance(row.get("nsi_reference"), dict) else {}
    return [float(nsi.get(f"{IDENTITY_PREFIX}{index}", 0.0) or 0.0) for index in range(4)]


def _set_scores(row: dict[str, Any], scores: list[float]) -> None:
    nsi = row.setdefault("nsi_reference", {})
    if not isinstance(nsi, dict):
        nsi = {}
        row["nsi_reference"] = nsi
    padded = list(scores[:4]) + [0.0] * max(0, 4 - len(scores))
    for index, value in enumerate(padded[:4]):
        nsi[f"{IDENTITY_PREFIX}{index}"] = float(value)
    ordered = sorted(padded[:4], reverse=True)
    confidence = ordered[0] if ordered else 0.0
    margin = confidence - (ordered[1] if len(ordered) > 1 else 0.0)
    nsi["command_identity_confidence"] = float(confidence)
    nsi["command_identity_margin"] = float(margin)


def _seed_for_row(row: dict[str, Any], seed: int) -> int:
    key = json.dumps(
        {
            "seed": seed,
            "example_id": row.get("example_id"),
            "episode_id": row.get("episode_id"),
            "slot": row.get("command_slot"),
            "count": _candidate_count(row),
        },
        sort_keys=True,
    )
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)


def _permutation(count: int, row: dict[str, Any], seed: int) -> list[int]:
    original = list(range(count))
    if count <= 1:
        return original
    rng = random.Random(_seed_for_row(row, seed))
    for _ in range(12):
        perm = original[:]
        rng.shuffle(perm)
        if perm != original:
            return perm
    return original[1:] + original[:1]


def _replace_neutral_candidate_sections(prompt: str, candidate_count: int) -> str:
    # The neutralized Phase2AN prompts contain identical candidate lines. Rewriting the
    # section keeps prompt/list count aligned after permutation without adding slot markers.
    repair_line = (
        "- repair_action=bounded_repair_action; intent=apply_patch_and_rerun_tests; "
        "edit_scope=bounded_public_source_patch; target_symbol=runtime_visible_symbol; "
        "verification_command=python -m pytest -q <generated_repair_test> --maxfail=1; "
        "description=Apply the bounded public repository repair and verify the generated failing test."
    )
    command_line = (
        "- bounded_repair_action intent=apply_patch_and_rerun_tests "
        "edit_scope=bounded_public_source_patch target_symbol=runtime_visible_symbol"
    )
    replacements = {
        "Candidate repair actions": "\n".join(repair_line for _ in range(candidate_count)),
        "Candidate commands": "\n".join(command_line for _ in range(candidate_count)),
    }
    output = prompt
    for heading, replacement in replacements.items():
        pattern = rf"({re.escape(heading)}:\n)(.*?)(?=\n\n[A-Z][^\n]+:\n|\n\nHead constraints:|\Z)"
        output, count = re.subn(pattern, rf"\1{replacement}", output, count=1, flags=re.S)
        if count != 1:
            raise ValueError(f"Could not replace prompt section: {heading}")
    return output


def permute_candidate_order(row: dict[str, Any], *, seed: int = 20260531) -> dict[str, Any]:
    count = _candidate_count(row)
    if count <= 1:
        raise ValueError("row must contain at least two candidates")
    old_slot = row.get("command_slot")
    if not isinstance(old_slot, int) or not 0 <= old_slot < count:
        raise ValueError("row has invalid command_slot")
    perm = _permutation(count, row, seed)
    out = json.loads(json.dumps(row))
    candidates = list(row.get("candidate_commands") or [])
    out["candidate_commands"] = [candidates[old_index] for old_index in perm]
    out["command_slot"] = perm.index(old_slot)
    scores = _scores(row)
    new_scores = [0.0] * 4
    for new_index, old_index in enumerate(perm):
        new_scores[new_index] = scores[old_index]
    _set_scores(out, new_scores)
    out["state_prompt"] = _replace_neutral_candidate_sections(
        str(out.get("state_prompt") or ""),
        count,
    )
    controls = list(out.get("runtime_overrides") or [])
    if "phase2ao_candidate_order_permuted" not in controls:
        controls.append("phase2ao_candidate_order_permuted")
    out["runtime_overrides"] = controls
    out["phase2ao_order_permutation"] = {
        "old_to_new": {str(old_index): perm.index(old_index) for old_index in perm},
        "new_to_old": {str(new_index): old_index for new_index, old_index in enumerate(perm)},
        "old_gold_slot": old_slot,
        "new_gold_slot": out["command_slot"],
    }
    return out


def build_phase2ao_order_permutation_controls(
    *,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    manifest_json: str | Path | None = None,
    seed: int = 20260531,
) -> dict[str, Any]:
    rows = _read_jsonl(input_jsonl)
    output_rows = [permute_candidate_order(row, seed=seed) for row in rows]
    output_hash = _write_jsonl(output_jsonl, output_rows)
    changed = sum(
        1
        for before, after in zip(rows, output_rows)
        if before.get("command_slot") != after.get("command_slot")
    )
    non_identity_permutations = sum(
        1
        for row in output_rows
        if any(
            int(new_index) != int(old_index)
            for new_index, old_index in row["phase2ao_order_permutation"]["new_to_old"].items()
        )
    )
    checks = {
        "row_count_preserved": len(rows) == len(output_rows) and len(rows) > 0,
        "all_rows_have_non_identity_permutation": non_identity_permutations == len(rows),
        "sealed_v3_absent": all(
            not (row.get("source_trace") or {}).get("sealed_v3_used") for row in output_rows
        ),
        "candidate_counts_preserved": all(
            _candidate_count(before) == _candidate_count(after)
            for before, after in zip(rows, output_rows)
        ),
    }
    manifest = {
        "artifact_family": "phase2ao_order_permutation_controls",
        "passed": all(checks.values()),
        "checks": checks,
        "input_jsonl": str(Path(input_jsonl)),
        "output_jsonl": str(Path(output_jsonl)),
        "output_sha256": output_hash,
        "row_count": len(output_rows),
        "changed_gold_slot_rows": changed,
        "non_identity_permutation_rows": non_identity_permutations,
        "seed": seed,
        "interpretation": (
            "Candidate order is deterministically permuted while command_slot and sidecar slot scores "
            "are remapped to the same semantic candidate. This checks whether the model follows the "
            "sidecar after position changes rather than fixed slot priors."
        ),
        "claim_boundary": "Diagnostic non-sealed order control only.",
    }
    if manifest_json:
        _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2AO candidate-order permutation control.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    manifest = build_phase2ao_order_permutation_controls(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        manifest_json=args.manifest_json,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not manifest["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
