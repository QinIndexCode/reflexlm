from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.runtime_evidence_audit import (
    ReflexCoreRuntimeEvidenceAuditConfig,
    audit_reflexcore_runtime_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit ReflexCore V0 runtime observation and prediction-error evidence.",
    )
    parser.add_argument("--matrix-report-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--min-profile-runs", type=int, default=9)
    parser.add_argument("--min-profile-pass-rate", type=float, default=1.0)
    parser.add_argument("--min-live-episode-count", type=int, default=15)
    parser.add_argument("--min-runtime-observation-steps", type=int, default=30)
    parser.add_argument("--min-changed-file-observation-steps", type=int, default=10)
    parser.add_argument("--min-terminal-observation-steps", type=int, default=10)
    parser.add_argument("--min-observed-prediction-error-examples", type=int, default=30)
    args = parser.parse_args()

    report = audit_reflexcore_runtime_evidence(
        ReflexCoreRuntimeEvidenceAuditConfig(
            matrix_report_json=Path(args.matrix_report_json),
            output_json=Path(args.output_json) if args.output_json else None,
            min_profile_runs=args.min_profile_runs,
            min_profile_pass_rate=args.min_profile_pass_rate,
            min_live_episode_count=args.min_live_episode_count,
            min_runtime_observation_steps=args.min_runtime_observation_steps,
            min_changed_file_observation_steps=args.min_changed_file_observation_steps,
            min_terminal_observation_steps=args.min_terminal_observation_steps,
            min_observed_prediction_error_examples=(
                args.min_observed_prediction_error_examples
            ),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
