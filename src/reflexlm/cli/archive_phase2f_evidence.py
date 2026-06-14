from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_REPORT_DIR = Path("artifacts/reports/phase2f_rich_latent_fusion_canary")
DEFAULT_PACKAGE_ROOT = Path("artifacts/packages/phase2f_rich_latent_fusion_nervous_canary")
DEFAULT_OUTPUT_DIR = Path("artifacts/archives/phase2f_rich_latent_fusion_20260517")
DEFAULT_PAPER = Path("paper_draft.md")
DEFAULT_EXTERNAL_REPORT_DIR = Path("artifacts/reports/phase2g_external_trace_v1")
DEFAULT_EXTERNAL_DATASET_DIR = Path("artifacts/datasets/phase2g_external_trace_v1")
DEFAULT_EXTERNAL_SEAL = Path("artifacts/control/external_trace_v1.sealed")
OPTIONAL_EXTERNAL_BUNDLES = [
    (
        Path("artifacts/reports/phase2g_external_trace_v1"),
        Path("artifacts/datasets/phase2g_external_trace_v1"),
        Path("artifacts/control/external_trace_v1.sealed"),
    ),
    (
        Path("artifacts/reports/phase2g_external_trace_v2_semantic_required"),
        Path("artifacts/datasets/phase2g_external_trace_v2_semantic_required"),
        Path("artifacts/control/external_trace_v2_semantic_required.sealed"),
    ),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _run_manifest_paths(report_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for eval_path in sorted(report_dir.glob("*.json")):
        try:
            payload = _load_json(eval_path)
        except json.JSONDecodeError:
            continue
        run_path = payload.get("run_path")
        if not run_path:
            continue
        manifest = Path(run_path) / "run_manifest.json"
        if manifest.exists():
            paths.append(manifest)
    return paths


def _optional_external_paths() -> list[Path]:
    paths: list[Path] = []
    for report_dir, dataset_dir, seal_path in OPTIONAL_EXTERNAL_BUNDLES:
        if report_dir.exists():
            paths.extend(sorted(report_dir.glob("*.json")))
            paths.extend(sorted(report_dir.glob("*.md")))
            paths.extend(_run_manifest_paths(report_dir))
        if dataset_dir.exists():
            for name in [
                "challenge.jsonl",
                "episode_metadata.json",
                "manifest.json",
                "leakage_audit.json",
                "semantic_nn_audit.json",
                "command_slot_overlap_audit.json",
                "semantic_necessity_audit.json",
                "sealed_config_hash",
            ]:
                path = dataset_dir / name
                if path.exists():
                    paths.append(path)
        if seal_path.exists():
            paths.append(seal_path)
    return paths


def collect_phase2f_archive_paths(
    *,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    package_root: str | Path = DEFAULT_PACKAGE_ROOT,
    paper_path: str | Path = DEFAULT_PAPER,
) -> list[Path]:
    report_path = Path(report_dir)
    package_path = Path(package_root)
    paper = Path(paper_path)
    required: list[Path] = [
        report_path / "phase2d_final_gate.json",
        report_path
        / "phase2f_rich_latent_fusion_canary_r16_alpha32_lr1e-4_len256_cap2048.training_summary.json",
        paper,
    ]
    required.extend(sorted(report_path.glob("*.json")))
    required.extend(sorted(report_path.glob("*.md")))
    required.extend(sorted(package_path.glob("*/native_nervous_package.json")))
    required.extend(_run_manifest_paths(report_path))
    required.extend(_optional_external_paths())

    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required Phase2F archive artifacts: "
            + ", ".join(str(path) for path in missing)
        )
    return sorted(set(path.resolve() for path in required), key=lambda path: str(path).lower())


def build_archive_manifest(
    *,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    package_root: str | Path = DEFAULT_PACKAGE_ROOT,
    paper_path: str | Path = DEFAULT_PAPER,
) -> dict[str, Any]:
    paths = collect_phase2f_archive_paths(
        report_dir=report_dir,
        package_root=package_root,
        paper_path=paper_path,
    )
    files = []
    for path in paths:
        stat = path.stat()
        files.append(
            {
                "source_path": str(path),
                "sha256": _sha256(path),
                "size_bytes": stat.st_size,
                "mtime_utc": stat.st_mtime,
            }
        )
    aggregate_hash = hashlib.sha256(
        json.dumps(files, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return {
        "archive_family": "phase2f_evidence_manifest",
        "archive_mode": "manifest_sha256_only",
        "artifact_count": len(files),
        "aggregate_sha256": aggregate_hash,
        "files": files,
    }


def write_archive_manifest(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    package_root: str | Path = DEFAULT_PACKAGE_ROOT,
    paper_path: str | Path = DEFAULT_PAPER,
) -> dict[str, Any]:
    manifest = build_archive_manifest(
        report_dir=report_dir,
        package_root=package_root,
        paper_path=paper_path,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive Phase2F evidence by SHA256 manifest.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--package-root", default=str(DEFAULT_PACKAGE_ROOT))
    parser.add_argument("--paper-path", default=str(DEFAULT_PAPER))
    args = parser.parse_args()
    manifest = write_archive_manifest(
        args.output_dir,
        report_dir=args.report_dir,
        package_root=args.package_root,
        paper_path=args.paper_path,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
