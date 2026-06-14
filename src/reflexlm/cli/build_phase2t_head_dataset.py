from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2s_head_dataset import phase2s_repair_trace_to_head_row


def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _sha256(value: Any) -> str:
    digest = hashlib.sha256()
    if isinstance(value, list):
        for row in value:
            digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()
    digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _command_identity_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins: list[float] = []
    zero_margin_examples: list[str] = []
    command_rows = 0
    for row in rows:
        if int(row.get("command_slot", -100)) == -100:
            continue
        command_rows += 1
        ref = row.get("nsi_reference") if isinstance(row.get("nsi_reference"), dict) else {}
        margin = float(ref.get("command_identity_margin", 0.0) or 0.0)
        margins.append(margin)
        if margin <= 0.0:
            zero_margin_examples.append(str(row.get("example_id") or row.get("episode_id") or ""))
    positive_rows = sum(1 for margin in margins if margin > 0.0)
    return {
        "command_slot_rows": command_rows,
        "positive_margin_rows": positive_rows,
        "zero_margin_rows": len(zero_margin_examples),
        "min_margin": min(margins) if margins else None,
        "margin_gate_passed": bool(margins) and positive_rows == len(margins),
        "zero_margin_examples": zero_margin_examples[:10],
    }


def phase2t_repair_trace_to_head_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("phase") != "Phase2T":
        raise ValueError(f"expected Phase2T trace row: {row.get('trace_id')}")
    if row.get("trace_construction_mode") != "phase2t_dynamic_public_repo_repair_loop_trace":
        raise ValueError(f"expected Phase2T dynamic repair trace: {row.get('trace_id')}")
    head_row = phase2s_repair_trace_to_head_row(row)
    original_prompt = str(head_row.get("state_prompt") or "")
    head_row["prompt_style"] = "phase2t_dynamic_repair_head_v1"
    head_row["state_prompt"] = original_prompt.replace(
        "Phase2S public repository repair native-head state input.",
        "Phase2T dynamic public repair-loop native-head state input.",
        1,
    ) + "\n\nPhase2T repair-loop constraints:\n- Patch proposal, test selection, rollback, and stop decisions must be grounded in recorded sandbox artifacts.\n- Modern agent-loop baselines must be measured separately; do not infer superiority from labels.\n"
    head_row["runtime_overrides"] = [
        "debug_cortex_escalation",
        "phase2t_dynamic_public_repair_loop",
        "sandboxed_patch_test_rollback_stop_evidence",
    ]
    head_row["source_trace"] = {
        **(head_row.get("source_trace") if isinstance(head_row.get("source_trace"), dict) else {}),
        "phase": "Phase2T",
        "trace_construction_mode": row.get("trace_construction_mode"),
        "repair_loop_schema": (row.get("repair_loop_episode") or {}).get("loop_schema")
        if isinstance(row.get("repair_loop_episode"), dict)
        else None,
        "claim_bearing_training_ready": row.get("claim_bearing_training_ready"),
        "sealed_v3_used": False,
    }
    return head_row


def build_phase2t_head_dataset(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path | None = None,
    output_dir: str | Path,
    data_health_json: str | Path | None = None,
    pretrain_gate_json: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    train_raw_rows = _read_jsonl(train_jsonl)
    val_raw_rows = _read_jsonl(val_jsonl)
    holdout_raw_rows = _read_jsonl(holdout_jsonl) if holdout_jsonl else []
    train_rows = [phase2t_repair_trace_to_head_row(row) for row in train_raw_rows]
    val_rows = [phase2t_repair_trace_to_head_row(row) for row in val_raw_rows]
    holdout_rows = [phase2t_repair_trace_to_head_row(row) for row in holdout_raw_rows]
    train_identity = _command_identity_diagnostics(train_rows)
    val_identity = _command_identity_diagnostics(val_rows)
    holdout_identity = _command_identity_diagnostics(holdout_rows) if holdout_jsonl else None
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "train.jsonl", train_rows)
    _write_jsonl(output / "val.jsonl", val_rows)
    if holdout_jsonl:
        _write_jsonl(output / "holdout.jsonl", holdout_rows)
    data_health = _read_json(data_health_json)
    pretrain_gate = _read_json(pretrain_gate_json)
    effective_split_hashes = (
        data_health.get("effective_split_hashes")
        if data_health
        else {
            "phase2t_train": _sha256(train_raw_rows),
            "phase2t_val": _sha256(val_raw_rows),
            **({"phase2t_holdout": _sha256(holdout_raw_rows)} if holdout_jsonl else {}),
        }
    )
    identity_diagnostics = {
        "train": train_identity,
        "val": val_identity,
    }
    if holdout_identity is not None:
        identity_diagnostics["holdout"] = holdout_identity
    all_identity_passed = train_identity["margin_gate_passed"] and val_identity[
        "margin_gate_passed"
    ]
    if holdout_identity is not None:
        all_identity_passed = all_identity_passed and holdout_identity["margin_gate_passed"]
    splits = {
        "train": {
            "source_jsonl": str(Path(train_jsonl)),
            "path": str(output / "train.jsonl"),
            "source_rows": len(train_raw_rows),
            "rows": len(train_rows),
            "source_sha256": _sha256(train_raw_rows),
            "sha256": _sha256(train_rows),
        },
        "val": {
            "source_jsonl": str(Path(val_jsonl)),
            "path": str(output / "val.jsonl"),
            "source_rows": len(val_raw_rows),
            "rows": len(val_rows),
            "source_sha256": _sha256(val_raw_rows),
            "sha256": _sha256(val_rows),
        },
    }
    if holdout_jsonl:
        splits["holdout"] = {
            "source_jsonl": str(Path(holdout_jsonl)),
            "path": str(output / "holdout.jsonl"),
            "source_rows": len(holdout_raw_rows),
            "rows": len(holdout_rows),
            "source_sha256": _sha256(holdout_raw_rows),
            "sha256": _sha256(holdout_rows),
        }
    manifest = {
        "dataset_family": "phase2t_dynamic_repair_head_dataset",
        "json_text_target": False,
        "sealed_v3_used": False,
        "claim_bearing_training_candidate": True,
        "source_data_health_passed": data_health.get("passed") if data_health else None,
        "source_pretrain_gate_passed": pretrain_gate.get("passed") if pretrain_gate else None,
        "command_identity_margin_gate_passed": all_identity_passed,
        "command_identity_diagnostics": identity_diagnostics,
        "effective_split_hashes": effective_split_hashes,
        "splits": splits,
        "inputs": {
            "data_health_json": str(Path(data_health_json)) if data_health_json else None,
            "pretrain_gate_json": str(Path(pretrain_gate_json)) if pretrain_gate_json else None,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build native-head training rows from Phase2T dynamic repair-loop traces."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--holdout-jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-health-json")
    parser.add_argument("--pretrain-gate-json")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    manifest = build_phase2t_head_dataset(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        holdout_jsonl=args.holdout_jsonl,
        output_dir=args.output_dir,
        data_health_json=args.data_health_json,
        pretrain_gate_json=args.pretrain_gate_json,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
