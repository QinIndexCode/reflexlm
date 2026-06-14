from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from reflexlm.llm.candidate_features import source_overlap_command_slot_prediction


NEUTRAL_COMMAND = (
    "bounded_repair_action intent=apply_patch_and_rerun_tests "
    "edit_scope=bounded_public_source_patch target_symbol=runtime_visible_symbol"
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _hash_rows(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _neutral_candidate_repair_actions(count: int) -> str:
    action = (
        "- repair_action=bounded_repair_action; intent=apply_patch_and_rerun_tests; "
        "edit_scope=bounded_public_source_patch; target_symbol=runtime_visible_symbol; "
        "verification_command=python -m pytest -q <generated_repair_test> --maxfail=1; "
        "description=Apply the bounded public repository repair and verify the generated failing test."
    )
    return "\n".join(action for _ in range(count))


def _neutral_candidate_commands(count: int) -> str:
    return "\n".join(f"- {NEUTRAL_COMMAND}" for _ in range(count))


def _replace_section(prompt: str, heading: str, replacement: str) -> str:
    pattern = rf"({re.escape(heading)}:\n)(.*?)(?=\n\n[A-Z][^\n]+:\n|\n\nHead constraints:|\Z)"
    replaced, count = re.subn(pattern, rf"\1{replacement}", prompt, count=1, flags=re.S)
    if count != 1:
        raise ValueError(f"Could not replace prompt section: {heading}")
    return replaced


def neutralize_candidate_text(row: dict[str, Any]) -> dict[str, Any]:
    candidate_count = len(row.get("candidate_commands") or [])
    if candidate_count <= 0:
        raise ValueError("row has no candidate_commands")
    out = dict(row)
    out["candidate_commands"] = [NEUTRAL_COMMAND for _ in range(candidate_count)]
    out["command"] = "bounded_repair_action"
    prompt = str(out.get("state_prompt") or "")
    prompt = _replace_section(
        prompt,
        "Candidate repair actions",
        _neutral_candidate_repair_actions(candidate_count),
    )
    prompt = _replace_section(prompt, "Candidate commands", _neutral_candidate_commands(candidate_count))
    out["state_prompt"] = prompt
    controls = list(out.get("runtime_overrides") or [])
    if "phase2an_candidate_text_neutralized" not in controls:
        controls.append("phase2an_candidate_text_neutralized")
    out["runtime_overrides"] = controls
    return out


def _contains_candidate_artifact(row: dict[str, Any]) -> bool:
    text = "\n".join(
        [
            " ".join(str(value) for value in row.get("candidate_commands") or []),
            str(row.get("state_prompt") or ""),
            str(row.get("command") or ""),
        ]
    )
    if re.search(r"repair_action_[0-9a-f]{8,}", text):
        return True
    symbols = re.findall(r"target_symbol=([A-Za-z_][A-Za-z0-9_]*)", text)
    return any(symbol != "runtime_visible_symbol" for symbol in symbols)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    correct = 0
    for row in rows:
        candidates = row.get("candidate_commands")
        slot = row.get("command_slot")
        if not isinstance(candidates, list) or not isinstance(slot, int):
            continue
        prediction = source_overlap_command_slot_prediction(str(row.get("state_prompt") or ""), candidates)
        total += 1
        correct += int(prediction == slot)
    return {
        "rows": len(rows),
        "rows_hash": _hash_rows(rows),
        "source_overlap_accuracy": correct / total if total else None,
        "source_overlap_correct": correct,
        "source_overlap_total": total,
        "candidate_count_distribution": {
            str(count): sum(1 for row in rows if len(row.get("candidate_commands") or []) == count)
            for count in sorted({len(row.get("candidate_commands") or []) for row in rows})
        },
        "candidate_artifact_rows": sum(1 for row in rows if _contains_candidate_artifact(row)),
    }


def build_phase2an_candidate_artifact_controls(
    *,
    original_jsonl: str | Path,
    erased_jsonl: str | Path,
    wrong_jsonl: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path | None = None,
) -> dict[str, Any]:
    sources = {
        "neutral_original": _read_jsonl(original_jsonl),
        "neutral_sidecar_erased": _read_jsonl(erased_jsonl),
        "neutral_wrong_sidecar": _read_jsonl(wrong_jsonl),
    }
    row_counts = {name: len(rows) for name, rows in sources.items()}
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"control row counts differ: {row_counts}")
    transformed = {
        name: [neutralize_candidate_text(row) for row in rows] for name, rows in sources.items()
    }
    output_root = Path(output_dir)
    outputs: dict[str, str] = {}
    for name, rows in transformed.items():
        out_path = output_root / f"{name}.jsonl"
        _write_jsonl(out_path, rows)
        outputs[name] = str(out_path)

    summaries = {name: _summarize(rows) for name, rows in transformed.items()}
    checks = {
        "row_counts_match": len(set(row_counts.values())) == 1,
        "candidate_artifacts_removed": all(
            summary["candidate_artifact_rows"] == 0 for summary in summaries.values()
        ),
        "source_overlap_nonzero_not_ceiling": all(
            isinstance(summary["source_overlap_accuracy"], int | float)
            and 0.0 < float(summary["source_overlap_accuracy"]) < 0.75
            for summary in summaries.values()
        ),
        "sealed_v3_absent": all(
            not row.get("source_trace", {}).get("sealed_v3_used")
            for rows in transformed.values()
            for row in rows
        ),
    }
    manifest = {
        "artifact_family": "phase2an_candidate_artifact_controls",
        "passed": all(checks.values()),
        "checks": checks,
        "input_jsonl": {
            "original": str(Path(original_jsonl)),
            "erased": str(Path(erased_jsonl)),
            "wrong": str(Path(wrong_jsonl)),
        },
        "outputs": outputs,
        "summaries": summaries,
        "interpretation": (
            "Candidate command text and prompt candidate sections are neutralized while preserving "
            "candidate count, order, labels, and sidecar condition. This isolates whether Phase2AM "
            "performance depends on candidate text/order artifacts rather than command-identity sidecar."
        ),
        "claim_boundary": (
            "This is a diagnostic control, not a new training source and not sealed evidence."
        ),
    }
    if manifest_json:
        _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2AN candidate artifact controls.")
    parser.add_argument("--original-jsonl", required=True)
    parser.add_argument("--erased-jsonl", required=True)
    parser.add_argument("--wrong-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    manifest = build_phase2an_candidate_artifact_controls(
        original_jsonl=args.original_jsonl,
        erased_jsonl=args.erased_jsonl,
        wrong_jsonl=args.wrong_jsonl,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not manifest["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
