from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_phase2w_reproduction_bundle(
    *,
    bundle_json: str | Path,
    run_commands: bool = False,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    bundle_path = Path(bundle_json)
    bundle = _read_json(bundle_path)
    root = Path(cwd) if cwd else Path.cwd()
    artifact_results = []
    for artifact in bundle.get("artifacts") or []:
        rel_path = Path(str(artifact.get("path") or ""))
        artifact_path = rel_path if rel_path.is_absolute() else root / rel_path
        actual = _sha256_file(artifact_path)
        expected = artifact.get("sha256")
        artifact_results.append(
            {
                "path": str(rel_path),
                "exists": artifact_path.exists(),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": actual == expected,
            }
        )
    command_results = []
    if run_commands:
        for command in bundle.get("verify_commands") or []:
            completed = subprocess.run(
                command,
                cwd=root,
                shell=True,
                text=True,
                capture_output=True,
            )
            command_results.append(
                {
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-2000:],
                    "passed": completed.returncode == 0,
                }
            )
    hashes_passed = bool(artifact_results) and all(
        item["matches"] for item in artifact_results
    )
    commands_passed = True if not run_commands else all(
        item["passed"] for item in command_results
    )
    return {
        "artifact_family": "phase2w_reproduction_bundle_verification",
        "passed": hashes_passed and commands_passed,
        "runner_independent": False,
        "verification_scope": "local_or_clean_runner_hash_and_command_verification_not_external_reproduction",
        "hashes_passed": hashes_passed,
        "commands_run": run_commands,
        "commands_passed": commands_passed,
        "artifact_results": artifact_results,
        "command_results": command_results,
        "blocked_actions": [
            "do_not_count_local_bundle_verification_as_independent_external_reproduction"
        ],
        "inputs": {"bundle_json": str(bundle_path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Phase2W reproduction bundle hashes and commands.")
    parser.add_argument("--bundle-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--run-commands", action="store_true")
    parser.add_argument("--cwd")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = verify_phase2w_reproduction_bundle(
        bundle_json=args.bundle_json,
        run_commands=args.run_commands,
        cwd=args.cwd,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
