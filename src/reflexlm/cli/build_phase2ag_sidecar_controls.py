from __future__ import annotations

import argparse
import hashlib
import json
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


def _valid_candidate_count(row: dict[str, Any]) -> int:
    candidates = row.get("candidate_commands")
    if isinstance(candidates, list):
        return min(4, max(0, len(candidates)))
    slot = row.get("command_slot")
    return min(4, max(0, int(slot) + 1 if isinstance(slot, int) else 4))


def _identity_scores(nsi: dict[str, Any]) -> list[float]:
    scores: list[float] = []
    for index in range(4):
        value = nsi.get(f"{IDENTITY_PREFIX}{index}")
        scores.append(float(value) if isinstance(value, (int, float)) else 0.0)
    return scores


def _set_identity_scores(nsi: dict[str, Any], scores: list[float]) -> None:
    padded = list(scores[:4]) + [0.0] * max(0, 4 - len(scores))
    for index in range(4):
        nsi[f"{IDENTITY_PREFIX}{index}"] = float(padded[index])
    sorted_scores = sorted(padded, reverse=True)
    confidence = sorted_scores[0] if sorted_scores else 0.0
    margin = confidence - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
    nsi["command_identity_confidence"] = float(confidence)
    nsi["command_identity_margin"] = float(margin)


def _erased(row: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(row))
    nsi = out.setdefault("nsi_reference", {})
    if isinstance(nsi, dict):
        _set_identity_scores(nsi, [0.0, 0.0, 0.0, 0.0])
    out["phase2ag_sidecar_control"] = "sidecar_erased"
    return out


def _wrong(row: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(row))
    nsi = out.setdefault("nsi_reference", {})
    if not isinstance(nsi, dict):
        out["phase2ag_sidecar_control"] = "wrong_sidecar"
        return out
    count = _valid_candidate_count(out)
    scores = _identity_scores(nsi)
    if count >= 2:
        valid = scores[:count]
        shifted = [valid[-1], *valid[:-1]]
        scores = shifted + [0.0] * (4 - count)
    else:
        scores = [0.0, 0.0, 0.0, 0.0]
    _set_identity_scores(nsi, scores)
    out["phase2ag_sidecar_control"] = "wrong_sidecar"
    return out


def build_phase2ag_sidecar_controls(
    *,
    input_jsonl: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path,
) -> dict[str, Any]:
    rows = _read_jsonl(input_jsonl)
    output = Path(output_dir)
    erased_rows = [_erased(row) for row in rows]
    wrong_rows = [_wrong(row) for row in rows]
    erased_path = output / "sidecar_erased.jsonl"
    wrong_path = output / "wrong_sidecar.jsonl"
    erased_hash = _write_jsonl(erased_path, erased_rows)
    wrong_hash = _write_jsonl(wrong_path, wrong_rows)

    wrong_changed = 0
    erased_changed = 0
    for original, erased, wrong in zip(rows, erased_rows, wrong_rows):
        original_scores = _identity_scores(original.get("nsi_reference") or {})
        erased_scores = _identity_scores(erased.get("nsi_reference") or {})
        wrong_scores = _identity_scores(wrong.get("nsi_reference") or {})
        if erased_scores != original_scores:
            erased_changed += 1
        if wrong_scores != original_scores:
            wrong_changed += 1

    report = {
        "artifact_family": "phase2ag_sidecar_controls",
        "passed": bool(rows) and erased_changed == len(rows) and wrong_changed == len(rows),
        "input_jsonl": str(Path(input_jsonl)),
        "row_count": len(rows),
        "controls": {
            "sidecar_erased": {
                "path": str(erased_path),
                "sha256": erased_hash,
                "changed_rows": erased_changed,
            },
            "wrong_sidecar": {
                "path": str(wrong_path),
                "sha256": wrong_hash,
                "changed_rows": wrong_changed,
            },
        },
        "blocked_actions": []
        if rows and erased_changed == len(rows) and wrong_changed == len(rows)
        else [
            "do_not_use_phase2ag_sidecar_controls",
            "do_not_claim_sidecar_dependence",
        ],
        "claim_boundary": (
            "These controls perturb only command_identity sidecar fields in an existing "
            "non-sealed head split. They test dependence on the sidecar, not sealed transfer "
            "or open-ended repair capability."
        ),
    }
    _write_json(manifest_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2AG sidecar-erased/wrong controls.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    args = parser.parse_args()
    report = build_phase2ag_sidecar_controls(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
