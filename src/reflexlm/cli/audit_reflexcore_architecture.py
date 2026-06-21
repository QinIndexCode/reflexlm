from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexlm.core.architecture_audit import (
    ReflexCoreArchitectureAuditConfig,
    audit_reflexcore_architecture,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit ReflexCore V0 structural architecture invariants.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-len", type=int, default=3)
    parser.add_argument("--min-parameter-count", type=int)
    parser.add_argument("--max-parameter-count", type=int)
    parser.add_argument("--min-numeric-action-aux-weight", type=float, default=0.0)
    parser.add_argument("--allow-missing-action-vector-residual", action="store_true")
    parser.add_argument("--allow-numeric-aux-text", action="store_true")
    parser.add_argument("--allow-numeric-aux-hash", action="store_true")
    args = parser.parse_args()

    report = audit_reflexcore_architecture(
        ReflexCoreArchitectureAuditConfig(
            config_path=Path(args.config),
            output_json=Path(args.output_json) if args.output_json else None,
            input_dim=args.input_dim,
            batch_size=args.batch_size,
            sequence_len=args.sequence_len,
            min_parameter_count=args.min_parameter_count,
            max_parameter_count=args.max_parameter_count,
            require_action_vector_residual=(
                not args.allow_missing_action_vector_residual
            ),
            min_numeric_action_aux_weight=args.min_numeric_action_aux_weight,
            require_numeric_aux_zero_text=not args.allow_numeric_aux_text,
            require_numeric_aux_zero_hash=not args.allow_numeric_aux_hash,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
