from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.action_conditioned_world_audit import (
    ReflexCoreActionConditionedWorldAuditConfig,
    audit_reflexcore_action_conditioned_world,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit ReflexCore V0 action-conditioned world-model paths.",
    )
    parser.add_argument("--output-json")
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--action-signal", type=float, default=4.0)
    parser.add_argument("--min-next-state-delta", type=float, default=1.0)
    parser.add_argument("--min-prediction-error-delta", type=float, default=0.5)
    args = parser.parse_args()

    report = audit_reflexcore_action_conditioned_world(
        ReflexCoreActionConditionedWorldAuditConfig(
            output_json=Path(args.output_json) if args.output_json else None,
            input_dim=args.input_dim,
            vocab_size=args.vocab_size,
            action_signal=args.action_signal,
            min_next_state_delta=args.min_next_state_delta,
            min_prediction_error_delta=args.min_prediction_error_delta,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
