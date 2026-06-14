from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "holdout")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _rows_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _repo_id(row: dict[str, Any]) -> str:
    origin = str(row.get("repo_url_or_origin") or row.get("repo_origin") or row.get("repo_id") or "repo")
    path = origin.removesuffix(".git").strip("/").split("/")
    if len(path) >= 2:
        return f"{path[-2]}_{path[-1]}".replace("-", "_")
    return path[-1].replace("-", "_") if path else "repo"


def _copy_artifact(
    *,
    source_dataset_root: Path,
    output_dataset_root: Path,
    source_rel: str,
    destination_rel: str,
) -> bool:
    source = source_dataset_root / source_rel
    if not source.exists() or not source.is_file():
        return False
    destination = output_dataset_root / destination_rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _source_patch_rel(row: dict[str, Any]) -> str:
    artifacts = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
    patch_rel = str(artifacts.get("patch_diff") or "")
    if patch_rel:
        return patch_rel.replace("\\", "/")
    generated = str(artifacts.get("generated_test") or "")
    if generated:
        return str(Path(generated).parent / "patch.diff").replace("\\", "/")
    return ""


def _rewrite_row(
    *,
    row: dict[str, Any],
    split: str,
    index: int,
    source_dataset_root: Path,
    output_dataset_root: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    artifacts = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
    generated_rel = str(artifacts.get("generated_test") or "").replace("\\", "/")
    patch_rel = _source_patch_rel(row)
    reasons: list[str] = []
    if not generated_rel:
        reasons.append("missing_generated_test_path")
    if not patch_rel:
        reasons.append("missing_patch_diff_path")
    if reasons:
        return None, reasons

    repo = _repo_id(row)
    dest_dir = Path("artifacts") / split / repo / f"row_{index:05d}"
    dest_generated = str(dest_dir / "generated_test.py").replace("\\", "/")
    dest_patch = str(dest_dir / "patch.diff").replace("\\", "/")
    if not _copy_artifact(
        source_dataset_root=source_dataset_root,
        output_dataset_root=output_dataset_root,
        source_rel=generated_rel,
        destination_rel=dest_generated,
    ):
        reasons.append("generated_test_copy_failed")
    if not _copy_artifact(
        source_dataset_root=source_dataset_root,
        output_dataset_root=output_dataset_root,
        source_rel=patch_rel,
        destination_rel=dest_patch,
    ):
        reasons.append("patch_diff_copy_failed")
    if reasons:
        return None, reasons

    converted = dict(row)
    converted["split"] = split
    converted["phase2aw_split_clean_artifact_rewrite"] = {
        "enabled": True,
        "source_generated_test": generated_rel,
        "source_patch_diff": patch_rel,
        "destination_generated_test": dest_generated,
        "destination_patch_diff": dest_patch,
    }
    converted["artifact_paths"] = {
        **artifacts,
        "generated_test": dest_generated,
        "patch_diff": dest_patch,
    }
    return converted, []


def build_phase2aw_split_clean_candidate_pool(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    source_dataset_root: str | Path,
    output_jsonl_dir: str | Path,
    output_dataset_root: str | Path,
    manifest_json: str | Path,
) -> dict[str, Any]:
    source_root = Path(source_dataset_root)
    output_root = Path(output_dataset_root)
    output_dir = Path(output_jsonl_dir)
    split_inputs = {
        "train": Path(train_jsonl),
        "val": Path(val_jsonl),
        "holdout": Path(holdout_jsonl),
    }
    split_rows: dict[str, list[dict[str, Any]]] = {}
    reject_counts: dict[str, int] = {}
    for split in SPLITS:
        converted_rows: list[dict[str, Any]] = []
        for row in _read_jsonl(split_inputs[split]):
            converted, reasons = _rewrite_row(
                row=row,
                split=split,
                index=len(converted_rows),
                source_dataset_root=source_root,
                output_dataset_root=output_root,
            )
            if converted is None:
                for reason in reasons:
                    reject_counts[reason] = reject_counts.get(reason, 0) + 1
                continue
            converted_rows.append(converted)
        split_rows[split] = converted_rows
        _write_jsonl(output_dir / f"{split}.jsonl", converted_rows)
    split_counts = {split: len(rows) for split, rows in split_rows.items()}
    passed = all(count > 0 for count in split_counts.values()) and not reject_counts
    manifest = {
        "artifact_family": "phase2aw_split_clean_candidate_pool",
        "passed": passed,
        "claim_boundary": (
            "Phase2AW split-clean candidate pool is a non-sealed data/runtime "
            "hardening artifact. It only proves artifact provenance hygiene; it "
            "does not authorize package, sealed evaluation, production autonomy, "
            "open-ended debugging generalization, or epoch-making claims."
        ),
        "source_dataset_root": str(source_root),
        "output_dataset_root": str(output_root),
        "output_jsonl_dir": str(output_dir),
        "split_inputs": {split: str(path) for split, path in split_inputs.items()},
        "split_counts": split_counts,
        "split_hashes": {
            split: _rows_sha256(rows) for split, rows in split_rows.items()
        },
        "reject_counts": dict(sorted(reject_counts.items())),
        "artifact_paths_rewritten": True,
        "source_artifact_split_clean_by_construction": True,
        "next_gate": "phase2aw_build_runtime_tasks_and_execution_gate",
        "unsupported_claims": [
            "phase2aw_package_ready",
            "sealed_cross_model_transfer",
            "learned_freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
    }
    _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AW split-clean candidate pool with copied patch/test artifacts."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--source-dataset-root", required=True)
    parser.add_argument("--output-jsonl-dir", required=True)
    parser.add_argument("--output-dataset-root", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2aw_split_clean_candidate_pool(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        holdout_jsonl=args.holdout_jsonl,
        source_dataset_root=args.source_dataset_root,
        output_jsonl_dir=args.output_jsonl_dir,
        output_dataset_root=args.output_dataset_root,
        manifest_json=args.manifest_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
