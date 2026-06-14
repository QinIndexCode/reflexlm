from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reflexlm.paper_tables import write_paper_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Paper B artifact-backed LaTeX tables.")
    parser.add_argument("--output-dir", default="docs/paper_b/tables")
    parser.add_argument("--manifest-json", default="docs/paper_b/tables/table_manifest.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    manifest = write_paper_tables(
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
        root=args.root,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
