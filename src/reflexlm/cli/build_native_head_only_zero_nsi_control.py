"""Build a native-head-only zero-NSI control split from a head dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTROL_OVERRIDE = "native_head_only_zero_nsi_control"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _runtime_overrides(value: Any) -> list[str]:
    if isinstance(value, list):
        overrides = [str(item) for item in value]
    elif value is None:
        overrides = []
    else:
        overrides = [str(value)]
    if CONTROL_OVERRIDE not in overrides:
        overrides.append(CONTROL_OVERRIDE)
    return overrides


def build_native_head_only_zero_nsi_control(
    *,
    source_jsonl: str | Path,
    output_jsonl: str | Path,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    source_path = Path(source_jsonl)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    source_rows_with_nsi_reference = 0
    output_rows: list[str] = []
    with source_path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            nsi_reference = row.get("nsi_reference")
            if isinstance(nsi_reference, dict) and nsi_reference:
                source_rows_with_nsi_reference += 1
            elif nsi_reference not in (None, {}, []):
                source_rows_with_nsi_reference += 1
            row["nsi_reference"] = {}
            row["runtime_overrides"] = _runtime_overrides(row.get("runtime_overrides"))
            output_rows.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
            rows += 1

    output_path.write_text("\n".join(output_rows) + ("\n" if output_rows else ""), encoding="utf-8")
    report = {
        "artifact_family": "native_head_only_zero_nsi_control_manifest",
        "source_jsonl": str(source_path),
        "output_jsonl": str(output_path),
        "rows": rows,
        "source_rows_with_nsi_reference": source_rows_with_nsi_reference,
        "nsi_reference_erased": True,
        "runtime_override_added": CONTROL_OVERRIDE,
        "sealed_v3_used_for_training_or_tuning": False,
        "sha256": _sha256(output_path),
    }
    if output_json is not None:
        output_report_path = Path(output_json)
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        output_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    report = build_native_head_only_zero_nsi_control(
        source_jsonl=args.source_jsonl,
        output_jsonl=args.output_jsonl,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
