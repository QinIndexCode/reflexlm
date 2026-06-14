from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def public_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(
        {
            line.strip().replace("\\", "/")
            for line in result.stdout.splitlines()
            if line.strip() and (ROOT / line.strip()).is_file()
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a clean, history-free public repository snapshot."
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    root = ROOT.resolve()
    if output == root or root in output.parents:
        raise SystemExit("Output must be outside the source repository.")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}")

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit-public-release.py")],
        cwd=ROOT,
        check=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    for relative in public_paths():
        source = ROOT / relative
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    print(f"Exported {len(public_paths())} files to {output}")
    print("Initialize a new Git repository in the exported directory.")


if __name__ == "__main__":
    main()
