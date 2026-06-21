from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.mechanism_dossier import verify_reflexcore_mechanism_dossier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a bounded ReflexCore V0 mechanism evidence dossier.",
    )
    parser.add_argument("--dossier-json", required=True)
    parser.add_argument("--base-dir")
    args = parser.parse_args()

    report = verify_reflexcore_mechanism_dossier(
        Path(args.dossier_json),
        base_dir=Path(args.base_dir) if args.base_dir else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
