from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.homeostatic_motor_audit import (
    ReflexCoreHomeostaticMotorAuditConfig,
    audit_reflexcore_homeostatic_motor,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit ReflexCore V0 homeostatic motor-head modulation.",
    )
    parser.add_argument("--output-json")
    parser.add_argument("--risk-block-threshold", type=float, default=0.9)
    parser.add_argument("--prediction-error-refresh-threshold", type=float, default=0.05)
    parser.add_argument(
        "--observed-prediction-error-refresh-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument("--salience-refresh-threshold", type=float, default=0.75)
    args = parser.parse_args()

    report = audit_reflexcore_homeostatic_motor(
        ReflexCoreHomeostaticMotorAuditConfig(
            output_json=Path(args.output_json) if args.output_json else None,
            risk_block_threshold=args.risk_block_threshold,
            prediction_error_refresh_threshold=args.prediction_error_refresh_threshold,
            observed_prediction_error_refresh_threshold=(
                args.observed_prediction_error_refresh_threshold
            ),
            salience_refresh_threshold=args.salience_refresh_threshold,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
