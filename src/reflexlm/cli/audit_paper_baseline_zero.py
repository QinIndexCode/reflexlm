from __future__ import annotations

import argparse
import json

from reflexlm.paper_baseline_audit import write_baseline_zero_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Paper B zero-valued baseline interpretability.")
    parser.add_argument(
        "--output-json",
        default="artifacts/reports/paper_b_baseline_zero_audit/baseline_zero_interpretability.json",
    )
    parser.add_argument(
        "--output-md",
        default="artifacts/reports/paper_b_baseline_zero_audit/baseline_zero_interpretability.md",
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    report = write_baseline_zero_audit(
        output_json=args.output_json,
        output_md=args.output_md,
        root=args.root,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
