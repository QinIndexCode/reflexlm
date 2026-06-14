from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.check_phase2d_gates import _metric, _trace_audit


def _load(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _completion(payload: dict[str, Any] | None) -> float | None:
    return _metric(payload, "task_completion_rate")


def build_external_gate_report(
    *,
    full_eval_json: str | Path,
    prompt_eval_json: str | Path,
    react_eval_json: str | Path,
    no_nsi_eval_json: str | Path,
    native_head_only_eval_json: str | Path,
    continuation_only_eval_json: str | Path,
    dataset_manifest_json: str | Path,
) -> dict[str, Any]:
    full = _load(full_eval_json)
    prompt = _load(prompt_eval_json)
    react = _load(react_eval_json)
    no_nsi = _load(no_nsi_eval_json)
    native = _load(native_head_only_eval_json)
    continuation = _load(continuation_only_eval_json)
    manifest = _load(dataset_manifest_json)

    full_completion = _completion(full)
    prompt_completion = _completion(prompt)
    react_completion = _completion(react)
    no_nsi_completion = _completion(no_nsi)
    native_completion = _completion(native)
    continuation_completion = _completion(continuation)
    best_text = max(
        [value for value in [prompt_completion, react_completion] if value is not None],
        default=None,
    )
    best_mechanism = max(
        [value for value in [native_completion, continuation_completion] if value is not None],
        default=None,
    )
    profile = str(manifest.get("profile") if manifest else "external_trace_v1")
    semantic_required = profile.startswith("external_trace_v") and profile.endswith(
        "_semantic_required"
    )
    full_trace = _trace_audit(full)
    full_hallucination = _metric(full, "state_hallucination_rate")
    mechanism_combination_gain = (
        full_completion is not None
        and best_mechanism is not None
        and full_completion > best_mechanism
    )
    mechanism_single_path_explains = (
        full_completion is not None
        and best_mechanism is not None
        and full_completion >= best_mechanism
        and not mechanism_combination_gain
    )
    checks = {
        "sealed_dataset": bool(manifest and manifest.get("sealed") and manifest.get("sealed_config_hash")),
        "full_completion": full_completion is not None and full_completion >= 0.90,
        "beats_text_baseline": best_text is not None
        and full_completion is not None
        and full_completion - best_text >= 0.30,
        "beats_no_nsi": no_nsi_completion is not None
        and full_completion is not None
        and full_completion - no_nsi_completion >= 0.15,
        "low_level_qwen_calls_zero": full_trace["low_level_qwen_calls"] == 0,
        "qwen_only_on_debug": full_trace["qwen_on_non_debug"] == 0,
        "allowlist_hallucination_zero": full_hallucination == 0.0,
    }
    if semantic_required:
        checks["beats_continuation_only"] = (
            continuation_completion is not None
            and full_completion is not None
            and full_completion - continuation_completion >= 0.15
        )
        checks["native_head_delta_or_native_head_suffices"] = (
            native_completion is not None
            and full_completion is not None
            and continuation_completion is not None
            and (
                full_completion - native_completion >= 0.10
                or (
                    native_completion >= 0.90
                    and native_completion - continuation_completion >= 0.15
                )
            )
        )
    else:
        checks["mechanism_delta_or_explained"] = (
            mechanism_combination_gain or mechanism_single_path_explains
        )
    passed = all(checks.values())
    return {
        "gate_family": f"{profile}_transfer_gate",
        "passed": passed,
        "checks": checks,
        "claim_scope": (
            "semantic_required_debug_cortex_supported"
            if semantic_required and passed
            else "semantic_required_debug_cortex_not_proven"
            if semantic_required
            else "combined_native_nervous_package"
            if mechanism_combination_gain
            else "single_mechanism_explains_external_result"
            if mechanism_single_path_explains
            else "failed_external_transfer"
        ),
        "metrics": {
            "profile": profile,
            "full_completion": full_completion,
            "prompt_completion": prompt_completion,
            "react_completion": react_completion,
            "best_text_completion": best_text,
            "no_nsi_completion": no_nsi_completion,
            "native_head_only_completion": native_completion,
            "continuation_only_completion": continuation_completion,
            "best_mechanism_completion": best_mechanism,
            "full_minus_best_text": (
                full_completion - best_text
                if full_completion is not None and best_text is not None
                else None
            ),
            "full_minus_no_nsi": (
                full_completion - no_nsi_completion
                if full_completion is not None and no_nsi_completion is not None
                else None
            ),
            "full_minus_continuation_only": (
                full_completion - continuation_completion
                if full_completion is not None and continuation_completion is not None
                else None
            ),
            "full_minus_native_head_only": (
                full_completion - native_completion
                if full_completion is not None and native_completion is not None
                else None
            ),
            "full_model_calls": _metric(full, "model_calls"),
            "full_state_hallucination": full_hallucination,
        },
        "trace_audit": {
            "full": full_trace,
            "prompt": _trace_audit(prompt),
            "react": _trace_audit(react),
            "no_nsi": _trace_audit(no_nsi),
            "native_head_only": _trace_audit(native),
            "continuation_only": _trace_audit(continuation),
        },
        "inputs": {
            "full_eval_json": str(Path(full_eval_json)),
            "prompt_eval_json": str(Path(prompt_eval_json)),
            "react_eval_json": str(Path(react_eval_json)),
            "no_nsi_eval_json": str(Path(no_nsi_eval_json)),
            "native_head_only_eval_json": str(Path(native_head_only_eval_json)),
            "continuation_only_eval_json": str(Path(continuation_only_eval_json)),
            "dataset_manifest_json": str(Path(dataset_manifest_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check sealed external trace transfer gate.")
    parser.add_argument("--full-eval-json", required=True)
    parser.add_argument("--prompt-eval-json", required=True)
    parser.add_argument("--react-eval-json", required=True)
    parser.add_argument("--no-nsi-eval-json", required=True)
    parser.add_argument("--native-head-only-eval-json", required=True)
    parser.add_argument("--continuation-only-eval-json", required=True)
    parser.add_argument("--dataset-manifest-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_external_gate_report(
        full_eval_json=args.full_eval_json,
        prompt_eval_json=args.prompt_eval_json,
        react_eval_json=args.react_eval_json,
        no_nsi_eval_json=args.no_nsi_eval_json,
        native_head_only_eval_json=args.native_head_only_eval_json,
        continuation_only_eval_json=args.continuation_only_eval_json,
        dataset_manifest_json=args.dataset_manifest_json,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
