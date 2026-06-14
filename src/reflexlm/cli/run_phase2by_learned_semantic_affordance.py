from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.baselines.state_rule_policy import BoundedStateRulePolicy
from reflexlm.cli.run_phase2bn_model_selected_sealed_runtime import (
    run_phase2bn_model_selected_sealed_runtime,
)
from reflexlm.cli.run_phase2br_runtime_ablation_comparison import _aggregate_subreports
from reflexlm.cli.run_phase2bx_semantic_signal_ablation import _manifest
from reflexlm.eval import EventGatedSequencePolicy
from reflexlm.models.semantic_matcher import HashedDualEncoderSemanticMatcher
from reflexlm.train import load_model_checkpoint


def _training_groups() -> dict[str, dict[str, list[str]]]:
    groups = {
        "dependency": {
            "observations": [
                "Module import cannot resolve a required library",
                "runtime lacks a required package",
                "dependency resolution failed",
                "import statement cannot load package",
            ],
            "commands": [
                "restore required library",
                "install missing dependency",
                "add required package",
            ],
        },
        "snapshot": {
            "observations": [
                "visual regression reference is stale",
                "rendered fixture differs from approved baseline",
                "golden file needs refresh",
            ],
            "commands": [
                "refresh golden reference",
                "update stored snapshot",
                "regenerate approved baseline",
            ],
        },
        "assertion": {
            "observations": [
                "test comparison produced an unexpected value",
                "verification expected one value but got another",
                "unit check failed its equality condition",
            ],
            "commands": [
                "rerun targeted unit test",
                "execute pytest verification",
                "inspect failing assertion",
            ],
        },
        "port": {
            "observations": [
                "listener cannot bind because endpoint is occupied",
                "service socket collision detected",
                "another process owns the network endpoint",
                "network address is busy",
            ],
            "commands": [
                "stop existing listener",
                "free network endpoint",
                "release occupied port",
            ],
        },
        "permission": {
            "observations": [
                "write operation lacks authorization",
                "filesystem operation is forbidden",
                "insufficient access rights",
                "access was denied by filesystem",
            ],
            "commands": [
                "grant required access",
                "repair filesystem permissions",
                "restore write authorization",
            ],
        },
        "path": {
            "observations": [
                "required folder cannot be found",
                "target location is absent",
                "filesystem location must be created",
                "file path is missing",
                "directory cannot be located",
            ],
            "commands": [
                "create required folder",
                "ensure target path exists",
                "materialize missing directory",
            ],
        },
        "dependency_path": {
            "observations": [
                "package cannot load and output folder is absent",
                "required library missing while destination path is absent",
                "dependency failure with missing filesystem location",
                "import unavailable and directory cannot be found",
            ],
            "commands": [
                "install dependency and create directory",
                "restore package and ensure path",
                "add library and materialize folder",
            ],
        },
        "port_permission": {
            "observations": [
                "network endpoint is occupied and write access is denied",
                "socket collision plus insufficient filesystem rights",
                "busy service port with forbidden file operation",
                "address in use while file access denied",
                "port conflict and permission denied",
            ],
            "commands": [
                "release port and repair permissions",
                "stop listener and grant file access",
                "free endpoint and restore authorization",
            ],
        },
    }
    for composite, left, right in [
        ("dependency_path", "dependency", "path"),
        ("port_permission", "port", "permission"),
    ]:
        groups[composite]["observations"].extend(
            f"{left_text} and {right_text}"
            for left_text in groups[left]["observations"][:3]
            for right_text in groups[right]["observations"][:3]
        )
        groups[composite]["commands"].extend(
            f"{left_text} and {right_text}"
            for left_text in groups[left]["commands"]
            for right_text in groups[right]["commands"]
        )
    return groups


def _policy_factories(
    *,
    checkpoint_path: str | Path,
    matcher: HashedDualEncoderSemanticMatcher,
) -> dict[str, Any]:
    model, vectorizer, checkpoint_payload = load_model_checkpoint(checkpoint_path, device="cpu")
    training_summary = checkpoint_payload.get("training_summary", {})
    common = {
        "model": model,
        "vectorizer": vectorizer,
        "training_summary": training_summary,
        "authorize_bounded_debug_cortex_recovery": True,
        "use_synaptic_motor_plan": True,
        "command_permutation_ensemble": True,
    }
    return {
        "learned_semantic_native": lambda: EventGatedSequencePolicy(
            policy_label="phase2by_learned_semantic_native",
            semantic_command_prior_weight=8.0,
            semantic_command_scorer=matcher,
            **common,
        ),
        "ontology_semantic_native": lambda: EventGatedSequencePolicy(
            policy_label="phase2by_ontology_semantic_native",
            semantic_command_prior_weight=1.0,
            **common,
        ),
        "equivariant_no_semantic_prior": lambda: EventGatedSequencePolicy(
            policy_label="phase2by_equivariant_no_semantic_prior",
            **common,
        ),
        "bounded_state_rule": lambda: BoundedStateRulePolicy(
            policy_label="phase2by_bounded_state_rule"
        ),
    }


def _run_suite(
    *,
    rows: list[dict[str, Any]],
    checkpoint_path: str | Path,
    matcher: HashedDualEncoderSemanticMatcher,
    output_root: Path,
    timeout_seconds: float,
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for policy_id, factory in _policy_factories(
        checkpoint_path=checkpoint_path,
        matcher=matcher,
    ).items():
        policy = factory()
        subreports = []
        for row in rows:
            repository_output = output_root / policy_id / row["repository_id"]
            subreports.append(
                run_phase2bn_model_selected_sealed_runtime(
                    checkpoint_path=None,
                    manifest_json=row["manifest_json"],
                    output_jsonl=repository_output / "trajectories.jsonl",
                    output_report_json=repository_output / "report.json",
                    timeout_seconds=timeout_seconds,
                    max_extra_steps=3,
                    policy_label=f"phase2by_{policy_id}",
                    policy_instance=policy,
                )
            )
        reports[policy_id] = {
            "metrics": _aggregate_subreports(subreports),
            "repository_report_jsons": [
                str(output_root / policy_id / row["repository_id"] / "report.json")
                for row in rows
            ],
        }
    return reports


def run_phase2by_learned_semantic_affordance(
    *,
    checkpoint_path: str | Path,
    phase2bq_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    phase2bq_report = json.loads(Path(phase2bq_report_json).read_text(encoding="utf-8-sig"))
    output_root = Path(output_dir)
    matcher = HashedDualEncoderSemanticMatcher(bins=2048, embedding_dim=96, seed=17)
    matcher_summary = matcher.fit(
        _training_groups(),
        epochs=2200,
        learning_rate=0.025,
    )
    matcher_path = matcher.save(output_root / "learned_semantic_matcher.pt")
    manifest_root = output_root / "manifests"
    suites: dict[str, list[dict[str, Any]]] = {"visible": [], "erased": []}
    for suite_id, erased in [("visible", False), ("erased", True)]:
        for repository in phase2bq_report["repository_reports"]:
            repository_id = str(repository["repository_id"])
            manifest = _manifest(
                workspace_root=repository["provenance"]["git_root"],
                repository_id=repository_id,
                erased=erased,
            )
            manifest_path = manifest_root / suite_id / f"{repository_id}.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            suites[suite_id].append(
                {"repository_id": repository_id, "manifest_json": str(manifest_path)}
            )
    visible_reports = _run_suite(
        rows=suites["visible"],
        checkpoint_path=checkpoint_path,
        matcher=matcher,
        output_root=output_root / "visible",
        timeout_seconds=timeout_seconds,
    )
    erased_reports = _run_suite(
        rows=suites["erased"],
        checkpoint_path=checkpoint_path,
        matcher=matcher,
        output_root=output_root / "erased",
        timeout_seconds=timeout_seconds,
    )
    learned_visible = visible_reports["learned_semantic_native"]["metrics"]
    learned_erased = erased_reports["learned_semantic_native"]["metrics"]
    ontology_visible = visible_reports["ontology_semantic_native"]["metrics"]
    no_prior_visible = visible_reports["equivariant_no_semantic_prior"]["metrics"]
    rule_visible = visible_reports["bounded_state_rule"]["metrics"]
    checks = {
        "phase2bq_source_report_passed": phase2bq_report.get("passed") is True,
        "matcher_training_top1_accuracy_is_complete": matcher_summary.training_top1_accuracy == 1.0,
        "learned_matcher_runtime_ontology_lookup_disabled": matcher.metadata()[
            "runtime_ontology_lookup"
        ]
        is False,
        "learned_semantic_native_solves_visible_holdout": learned_visible[
            "task_completion_success_rate"
        ]
        == 1.0,
        "learned_semantic_signal_erasure_reduces_completion": learned_erased[
            "task_completion_success_rate"
        ]
        < learned_visible["task_completion_success_rate"],
        "learned_semantic_outperforms_no_prior": learned_visible[
            "task_completion_success_rate"
        ]
        > no_prior_visible["task_completion_success_rate"],
        "learned_semantic_outperforms_state_rule": learned_visible[
            "task_completion_success_rate"
        ]
        > rule_visible["task_completion_success_rate"],
        "learned_semantic_matches_ontology_holdout_completion": learned_visible[
            "task_completion_success_rate"
        ]
        == ontology_visible["task_completion_success_rate"],
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2by_learned_semantic_affordance",
        "passed": passed,
        "ready_for_bounded_learned_semantic_affordance_claim": passed,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checkpoint_path": str(Path(checkpoint_path)),
        "matcher_path": str(matcher_path),
        "matcher_metadata": matcher.metadata(),
        "checks": checks,
        "visible_policy_reports": visible_reports,
        "erased_policy_reports": erased_reports,
        "comparison": {
            "learned_visible_completion": learned_visible["task_completion_success_rate"],
            "learned_erased_completion": learned_erased["task_completion_success_rate"],
            "ontology_visible_completion": ontology_visible["task_completion_success_rate"],
            "no_prior_visible_completion": no_prior_visible["task_completion_success_rate"],
            "rule_visible_completion": rule_visible["task_completion_success_rate"],
        },
        "supported_claims": [
            "a learned bounded dual-encoder matcher selected paraphrased and compositional recovery candidates without runtime ontology lookup and degraded under semantic signal erasure"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "open-vocabulary semantic affordance discovery",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2bz_open_vocabulary_semantic_affordance_transfer"
            if passed
            else "repair_phase2by_learned_semantic_affordance"
        ),
    }
    output_path = Path(output_report_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate an ontology-free runtime semantic matcher."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--phase2bq-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2by_learned_semantic_affordance(
        checkpoint_path=args.checkpoint_path,
        phase2bq_report_json=args.phase2bq_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
