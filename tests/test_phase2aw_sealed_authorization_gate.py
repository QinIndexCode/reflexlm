import json
from pathlib import Path

from reflexlm.cli.audit_phase2aw_sealed_authorization_gate import (
    audit_phase2aw_sealed_authorization_gate,
)
from reflexlm.llm.native_nervous_package import write_native_nervous_package


def _write(path: Path, payload: dict | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sufficiency(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "sufficiency.json",
        {
            "passed": True,
            "claim_scope": "phase2aw_bounded_nonsealed_package_loaded_descriptor_runtime",
            "unsupported_claims": ["epoch_making_architecture"],
        },
    )


def _package(
    tmp_path: Path,
    label: str,
    *,
    zero_nsi: bool = False,
    continuation: bool = True,
    native_heads: bool = True,
) -> Path:
    package = tmp_path / label
    write_native_nervous_package(
        package,
        base_model_name="qwen",
        native_head_path="heads",
        low_level_checkpoint_path="low.pt",
        policy_label=label,
        zero_nsi_latent=zero_nsi,
        continuation_cache_enabled=continuation,
        native_head_calls_enabled=native_heads,
    )
    return package


def _packages(tmp_path: Path) -> dict[str, Path]:
    label = "phase2aw_full"
    return {
        "full": _package(tmp_path, label),
        "no_nsi": _package(tmp_path, f"{label}_no_nsi_latent", zero_nsi=True),
        "native": _package(tmp_path, f"{label}_native_head_only", continuation=False),
        "continuation": _package(
            tmp_path,
            f"{label}_continuation_only",
            native_heads=False,
        ),
    }


def test_phase2aw_sealed_authorization_gate_accepts_matching_control_packages(
    tmp_path: Path,
) -> None:
    packages = _packages(tmp_path)
    sealed = _write(tmp_path / "sealed" / "challenge.jsonl", "{}\n")

    report = audit_phase2aw_sealed_authorization_gate(
        package_loaded_sufficiency_json=_sufficiency(tmp_path),
        full_package_path=packages["full"],
        no_nsi_package_path=packages["no_nsi"],
        native_head_only_package_path=packages["native"],
        continuation_only_package_path=packages["continuation"],
        sealed_dataset_path=sealed,
    )

    assert report["passed"] is True
    assert report["ready_for_sealed_eval"] is True
    assert report["ready_for_claim_upgrade"] is False
    assert "epoch_making_architecture_claim_not_established" in report["unsupported_claims"]


def test_phase2aw_sealed_authorization_gate_rejects_no_nsi_identity_leak(
    tmp_path: Path,
) -> None:
    packages = _packages(tmp_path)
    manifest = packages["no_nsi"] / "native_nervous_package.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["disabled_command_candidate_feature_groups"] = []
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    sealed = _write(tmp_path / "sealed" / "challenge.jsonl", "{}\n")

    report = audit_phase2aw_sealed_authorization_gate(
        package_loaded_sufficiency_json=_sufficiency(tmp_path),
        full_package_path=packages["full"],
        no_nsi_package_path=packages["no_nsi"],
        native_head_only_package_path=packages["native"],
        continuation_only_package_path=packages["continuation"],
        sealed_dataset_path=sealed,
    )

    assert report["passed"] is False
    assert report["checks"]["no_nsi_disables_candidate_identity"] is False
    assert "do_not_evaluate_no_nsi_with_candidate_identity_leak" in report["blocked_actions"]


def test_phase2aw_sealed_authorization_gate_rejects_control_label_mismatch(
    tmp_path: Path,
) -> None:
    packages = _packages(tmp_path)
    bad_native = _package(tmp_path, "phase2aw_other_native_head_only", continuation=False)
    sealed = _write(tmp_path / "sealed" / "challenge.jsonl", "{}\n")

    report = audit_phase2aw_sealed_authorization_gate(
        package_loaded_sufficiency_json=_sufficiency(tmp_path),
        full_package_path=packages["full"],
        no_nsi_package_path=packages["no_nsi"],
        native_head_only_package_path=bad_native,
        continuation_only_package_path=packages["continuation"],
        sealed_dataset_path=sealed,
    )

    assert report["passed"] is False
    assert report["checks"]["control_labels_derive_from_full_label"] is False
    assert "do_not_run_sealed_eval_with_mismatched_control_labels" in report["blocked_actions"]
