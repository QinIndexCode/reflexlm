import json
import hashlib
import subprocess
from pathlib import Path

from reflexlm.cli.audit_phase2m_external_generalization import (
    build_phase2m_data_health,
    build_phase2m_head_state_baseline_audit,
    build_phase2m_pretrain_gate,
    compute_phase2m_baseline_predictions,
)
from reflexlm.cli.audit_phase2m_v2_postflight import build_phase2m_v2_postflight
from reflexlm.cli.build_phase2m_head_dataset import (
    build_phase2m_head_dataset,
    phase2m_trace_to_head_row,
)
from reflexlm.cli.collect_phase2m_external_traces import normalize_phase2m_rows
from reflexlm.cli.collect_phase2m_public_repo_traces import (
    collect_phase2m_public_repo_traces,
)
from reflexlm.cli.check_phase2m_external_generalization import (
    build_phase2m_default_preregistration,
    build_phase2m_preregistration_check,
)
from reflexlm.cli.generate_phase2m_synthetic_safe_traces import (
    build_phase2m_synthetic_safe_rows,
    build_phase2m_v2_synthetic_safe_rows,
)
from reflexlm.cli.review_phase2m_design_maturity import (
    build_phase2m_design_maturity_review,
)
from reflexlm.cli.write_phase2m_v2_boundary import build_phase2m_v2_boundary


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _hash(text: str, length: int = 64) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _make_public_pytest_repo(tmp_path: Path, repo_id: str) -> Path:
    repo = tmp_path / repo_id
    (repo / "src" / repo_id).mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "LICENSE").write_text("MIT test fixture\n", encoding="utf-8")
    (repo / "src" / repo_id / "__init__.py").write_text("", encoding="utf-8")
    (repo / "src" / repo_id / "engine.py").write_text(
        "\n".join(
            [
                "def parse_record(value):",
                "    return str(value).strip()",
                "",
                "def render_packet(value):",
                "    return {'packet': value}",
                "",
                "def schedule_retry(value):",
                "    return value",
            ]
        ),
        encoding="utf-8",
    )
    functions = []
    names = [
        "test_accepts_trimmed_record",
        "test_rejects_empty_record",
        "test_preserves_unicode_packet",
        "test_renders_nested_packet",
        "test_retries_timeout_path",
        "test_schedules_backoff_path",
        "test_keeps_metadata_order",
        "test_handles_missing_owner",
        "test_reports_parser_context",
        "test_batches_retry_window",
        "test_merges_existing_payload",
        "test_formats_error_context",
    ]
    for name in names:
        functions.append(
            "\n".join(
                [
                    f"def {name}():",
                    "    observed = parse_record('  payload  ')",
                    "    packet = render_packet(observed)",
                    "    retry = schedule_retry(packet)",
                    "    assert retry",
                    "",
                ]
            )
        )
    (repo / "tests" / "test_engine_flow.py").write_text(
        "from src.%s.engine import parse_record, render_packet, schedule_retry\n\n%s"
        % (repo_id, "\n".join(functions)),
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "phase2m@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Phase2M Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return repo


def _phase2m_rows(
    split: str,
    *,
    repo_prefix: str,
    count: int,
    source_solved: bool = False,
    native_solved: bool = False,
) -> list[dict]:
    densities = ["low", "medium", "high"]
    candidate_counts = [2, 3, 4]
    depths = ["one_step", "two_step", "stale_state_refresh"]
    ambiguity_classes = ["same_intent_command", "same_file_read", "stage_transition"]
    trace_types = [
        "test_failure_traceback_to_symbol",
        "changed_file_to_watched_test",
        "module_ownership_to_command",
        "stale_state_refresh",
    ]
    rows: list[dict] = []
    for index in range(count):
        candidate_count = candidate_counts[index % len(candidate_counts)]
        repo_id = f"{repo_prefix}-{index % 3}"
        commands = [
            {
                "command": f"python -m pytest tests/{repo_id}/test_case_{index}_{slot}.py",
                "intent": "test_rerun",
            }
            for slot in range(candidate_count)
        ]
        expected_slot = index % candidate_count
        source_slot = expected_slot if source_solved else (expected_slot + 1) % candidate_count
        native_slot = expected_slot if native_solved else (expected_slot + 1) % candidate_count
        expected_command = commands[expected_slot]["command"]
        rows.append(
            {
                "trace_id": f"{split}-{repo_id}-{index}",
                "trace_hash": _hash(f"{split}-{repo_id}-{index}", 40),
                "split": split,
                "source_kind": "synthetic_safe_repo",
                "repo_id": repo_id,
                "repo_url_or_origin": f"synthetic://phase2m/{repo_id}",
                "commit_hash": _hash(f"commit-{repo_id}", 40),
                "license_or_synthetic_origin": "synthetic-safe phase2m pressure fixture",
                "collection_script_hash": _hash("phase2m-readonly-collector"),
                "normalization": {
                    "deterministic": True,
                    "redacted_absolute_local_paths": True,
                    "redacted_secrets_tokens_and_emails": True,
                    "preserved_runtime_visible_evidence": True,
                },
                "current_visible_text": (
                    f"Traceback points to module_{index}.symbol; watched test group "
                    f"{repo_id}; changed file src/{repo_id}/module_{index}.py"
                ),
                "runtime_visible_evidence": {
                    "traceback_symbols": [f"module_{index}.symbol"],
                    "changed_files": [f"src/{repo_id}/module_{index}.py"],
                    "watched_files": [f"tests/{repo_id}/test_case_{index}.py"],
                    "module_owner": f"{repo_id}.owner",
                },
                "command_candidates": commands,
                "expected_command": expected_command,
                "baselines": {
                    "source_overlap": commands[source_slot]["command"],
                    "native_head_only": commands[native_slot]["command"],
                    "continuation_only": commands[(expected_slot + 1) % candidate_count][
                        "command"
                    ],
                    "prompt_only": commands[(expected_slot + 1) % candidate_count]["command"],
                    "react": commands[(expected_slot + 1) % candidate_count]["command"],
                },
                "difficulty": {
                    "evidence_density": densities[index % len(densities)],
                    "candidate_count": candidate_count,
                    "continuation_depth": depths[index % len(depths)],
                    "ambiguity_class": ambiguity_classes[index % len(ambiguity_classes)],
                    "trace_type": trace_types[index % len(trace_types)],
                },
            }
        )
    return rows


def test_phase2m_preregistration_accepts_readonly_public_trace_plan(tmp_path: Path) -> None:
    proposal = _write(tmp_path / "proposal.json", build_phase2m_default_preregistration())

    report = build_phase2m_preregistration_check(proposal_json=proposal)

    assert report["passed"] is True
    assert report["next_action"] == "collect_readonly_phase2m_trace_and_run_data_health_only"
    assert report["checks"]["trace_collection_read_only"] is True
    assert report["checks"]["no_gold_hidden_or_sealed_feedback"] is True


def test_phase2m_preregistration_rejects_sealed_training_feedback(tmp_path: Path) -> None:
    payload = build_phase2m_default_preregistration()
    payload["data_policy"]["training_roots"] = [
        "artifacts/datasets/phase2i_external_trace_v3_semantic_required"
    ]
    payload["data_policy"]["uses_sealed_failures_for_analysis_feedback"] = True
    proposal = _write(tmp_path / "proposal.json", payload)

    report = build_phase2m_preregistration_check(proposal_json=proposal)

    assert report["passed"] is False
    assert "do_not_use_gold_hidden_or_sealed_feedback" in report["blocked_actions"]


def test_phase2m_preregistration_rejects_write_side_effect_collection(
    tmp_path: Path,
) -> None:
    payload = build_phase2m_default_preregistration()
    payload["trace_policy"]["collection_mode"] = "run_tests_and_modify_repo"
    proposal = _write(tmp_path / "proposal.json", payload)

    report = build_phase2m_preregistration_check(proposal_json=proposal)

    assert report["passed"] is False
    assert "do_not_collect_phase2m_trace_with_write_side_effects" in report[
        "blocked_actions"
    ]


def test_phase2m_preregistration_rejects_private_secret_dependent_repos(
    tmp_path: Path,
) -> None:
    payload = build_phase2m_default_preregistration()
    payload["trace_policy"]["private_repo_allowed"] = True
    payload["trace_policy"]["requires_network_or_secrets"] = True
    proposal = _write(tmp_path / "proposal.json", payload)

    report = build_phase2m_preregistration_check(proposal_json=proposal)

    assert report["passed"] is False
    assert "do_not_use_private_or_secret_dependent_repos" in report["blocked_actions"]


def test_phase2m_preregistration_rejects_training_before_data_health(
    tmp_path: Path,
) -> None:
    payload = build_phase2m_default_preregistration()
    payload["execution_plan"]["starts_training"] = True
    proposal = _write(tmp_path / "proposal.json", payload)

    report = build_phase2m_preregistration_check(proposal_json=proposal)

    assert report["passed"] is False
    assert "do_not_start_phase2m_training_before_data_health" in report[
        "blocked_actions"
    ]


def test_phase2m_preregistration_requires_redaction_and_provenance(
    tmp_path: Path,
) -> None:
    payload = build_phase2m_default_preregistration()
    payload["trace_policy"]["records_commit_hash"] = False
    payload["trace_policy"]["redacts_secrets_tokens_and_emails"] = False
    proposal = _write(tmp_path / "proposal.json", payload)

    report = build_phase2m_preregistration_check(proposal_json=proposal)

    assert report["passed"] is False
    assert "do_not_generate_phase2m_data_without_provenance" in report["blocked_actions"]
    assert "do_not_generate_phase2m_data_without_redaction" in report["blocked_actions"]


def test_phase2m_data_health_accepts_readonly_pressure_traces(tmp_path: Path) -> None:
    train = _write_jsonl(
        tmp_path / "train.jsonl", _phase2m_rows("train", repo_prefix="train", count=24)
    )
    val = _write_jsonl(
        tmp_path / "val.jsonl", _phase2m_rows("val", repo_prefix="val", count=24)
    )
    holdout = _write_jsonl(
        tmp_path / "holdout.jsonl",
        _phase2m_rows("holdout", repo_prefix="holdout", count=12),
    )

    report = build_phase2m_data_health(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
    )

    assert report["passed"] is True
    assert report["allowed_next_action"] == "run_phase2m_smoke_training_only"
    assert report["checks"]["phase2m_repo_disjoint_holdout"] is True
    assert report["checks"]["phase2m_source_overlap_val_below_threshold"] is True
    assert report["checks"]["phase2m_native_head_only_val_below_threshold"] is True
    assert report["effective_split_hashes"]["phase2m_train"]


def test_phase2m_pretrain_gate_allows_only_after_data_health_passes(
    tmp_path: Path,
) -> None:
    train = _write_jsonl(
        tmp_path / "train.jsonl", _phase2m_rows("train", repo_prefix="train", count=24)
    )
    val = _write_jsonl(
        tmp_path / "val.jsonl", _phase2m_rows("val", repo_prefix="val", count=24)
    )
    holdout = _write_jsonl(
        tmp_path / "holdout.jsonl",
        _phase2m_rows("holdout", repo_prefix="holdout", count=12),
    )
    data_health = _write(
        tmp_path / "data_health.json",
        build_phase2m_data_health(
            train_jsonl=train,
            val_jsonl=val,
            holdout_jsonl=holdout,
        ),
    )

    report = build_phase2m_pretrain_gate(data_health_json=data_health)

    assert report["passed"] is True
    assert report["allowed_next_action"] == "run_phase2m_smoke_training_only"


def test_phase2m_pretrain_gate_rejects_head_state_source_overlap_shortcut(
    tmp_path: Path,
) -> None:
    train_rows = _phase2m_rows("train", repo_prefix="train", count=24)
    val_rows = _phase2m_rows("val", repo_prefix="val", count=24)
    for row in val_rows:
        row["runtime_visible_evidence"]["shortcut_visible_text"] = row["expected_command"]
    train = _write_jsonl(tmp_path / "train.jsonl", train_rows)
    val = _write_jsonl(tmp_path / "val.jsonl", val_rows)
    holdout = _write_jsonl(
        tmp_path / "holdout.jsonl",
        _phase2m_rows("holdout", repo_prefix="holdout", count=12),
    )
    data_health = _write(
        tmp_path / "data_health.json",
        build_phase2m_data_health(
            train_jsonl=train,
            val_jsonl=val,
            holdout_jsonl=holdout,
        ),
    )
    build_phase2m_head_dataset(
        train_jsonl=train,
        val_jsonl=val,
        output_dir=tmp_path / "heads",
    )
    head_state_baseline = _write(
        tmp_path / "head_state_baseline.json",
        build_phase2m_head_state_baseline_audit(
            head_train_jsonl=tmp_path / "heads" / "train.jsonl",
            head_val_jsonl=tmp_path / "heads" / "val.jsonl",
            max_val_source_overlap_accuracy=0.30,
        ),
    )

    report = build_phase2m_pretrain_gate(
        data_health_json=data_health,
        head_state_baseline_json=head_state_baseline,
    )

    assert report["passed"] is False
    assert report["checks"]["head_state_source_overlap_below_threshold"] is False
    assert "do_not_train_when_head_state_source_overlap_solves_phase2m_val" in report[
        "blocked_actions"
    ]


def test_phase2m_data_health_rejects_source_overlap_or_native_solved_val(
    tmp_path: Path,
) -> None:
    train = _write_jsonl(
        tmp_path / "train.jsonl", _phase2m_rows("train", repo_prefix="train", count=24)
    )
    val = _write_jsonl(
        tmp_path / "val.jsonl",
        _phase2m_rows(
            "val",
            repo_prefix="val",
            count=24,
            source_solved=True,
            native_solved=True,
        ),
    )
    holdout = _write_jsonl(
        tmp_path / "holdout.jsonl",
        _phase2m_rows("holdout", repo_prefix="holdout", count=12),
    )

    report = build_phase2m_data_health(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
    )

    assert report["passed"] is False
    assert "do_not_train_when_source_overlap_solves_phase2m_val" in report[
        "blocked_actions"
    ]
    assert "do_not_train_when_native_head_only_solves_phase2m_val" in report[
        "blocked_actions"
    ]


def test_phase2m_data_health_rejects_unredacted_or_sealed_visible_trace(
    tmp_path: Path,
) -> None:
    train_rows = _phase2m_rows("train", repo_prefix="train", count=24)
    val_rows = _phase2m_rows("val", repo_prefix="val", count=24)
    val_rows[0]["current_visible_text"] = (
        r"Traceback at C:\Users\Admin\secret.py token=abc123456789"
    )
    val_rows[1]["runtime_visible_evidence"]["sealed_feedback"] = (
        "external_trace_v3_semantic_required failure"
    )
    train = _write_jsonl(tmp_path / "train.jsonl", train_rows)
    val = _write_jsonl(tmp_path / "val.jsonl", val_rows)
    holdout = _write_jsonl(
        tmp_path / "holdout.jsonl",
        _phase2m_rows("holdout", repo_prefix="holdout", count=12),
    )

    report = build_phase2m_data_health(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
    )

    assert report["passed"] is False
    assert "do_not_train_with_unredacted_phase2m_traces" in report["blocked_actions"]
    assert "do_not_use_sealed_or_sealed_failure_feedback" in report["blocked_actions"]


def test_phase2m_data_health_rejects_non_disjoint_or_duplicate_holdout(
    tmp_path: Path,
) -> None:
    train_rows = _phase2m_rows("train", repo_prefix="shared", count=24)
    holdout_rows = _phase2m_rows("holdout", repo_prefix="shared", count=12)
    holdout_rows[0]["trace_hash"] = train_rows[0]["trace_hash"]
    holdout_rows[0]["commit_hash"] = train_rows[0]["commit_hash"]
    holdout_rows[0]["repo_id"] = train_rows[0]["repo_id"]
    train = _write_jsonl(tmp_path / "train.jsonl", train_rows)
    val = _write_jsonl(
        tmp_path / "val.jsonl", _phase2m_rows("val", repo_prefix="val", count=24)
    )
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", holdout_rows)

    report = build_phase2m_data_health(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
    )

    assert report["passed"] is False
    assert "do_not_train_without_repo_disjoint_holdout" in report["blocked_actions"]
    assert report["checks"]["phase2m_deduplicated_by_repo_commit_trace_hash"] is False


def test_phase2m_trace_normalizer_redacts_and_feeds_data_health(
    tmp_path: Path,
) -> None:
    train_rows = _phase2m_rows("train", repo_prefix="train", count=24)
    val_rows = _phase2m_rows("val", repo_prefix="val", count=24)
    holdout_rows = _phase2m_rows("holdout", repo_prefix="holdout", count=12)
    train_rows[0]["current_visible_text"] += (
        r" local path C:\Users\Admin\repo\tests\test_demo.py token=abc123456789"
    )
    train_rows[0].pop("trace_hash")
    train_rows[0].pop("collection_script_hash")
    train_rows[0].pop("normalization")

    train = _write_jsonl(
        tmp_path / "train.jsonl",
        normalize_phase2m_rows(
            train_rows,
            split="train",
            collection_script_hash=_hash("normalizer-test-script"),
        ),
    )
    val = _write_jsonl(
        tmp_path / "val.jsonl",
        normalize_phase2m_rows(
            val_rows,
            split="val",
            collection_script_hash=_hash("normalizer-test-script"),
        ),
    )
    holdout = _write_jsonl(
        tmp_path / "holdout.jsonl",
        normalize_phase2m_rows(
            holdout_rows,
            split="holdout",
            collection_script_hash=_hash("normalizer-test-script"),
        ),
    )
    normalized_train = [
        json.loads(line)
        for line in train.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert "C:\\Users" not in normalized_train[0]["current_visible_text"]
    assert "abc123456789" not in normalized_train[0]["current_visible_text"]
    assert normalized_train[0]["normalization"]["deterministic"] is True

    report = build_phase2m_data_health(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
    )

    assert report["passed"] is True


def test_phase2m_synthetic_safe_generator_covers_registered_dimensions() -> None:
    rows = build_phase2m_synthetic_safe_rows(
        split="val",
        count=24,
        repo_prefix="val",
    )

    assert {row["difficulty"]["evidence_density"] for row in rows} == {
        "low",
        "medium",
        "high",
    }
    assert {row["difficulty"]["candidate_count"] for row in rows} == {2, 3, 4}
    assert {
        row["difficulty"]["continuation_depth"] for row in rows
    } == {"one_step", "two_step", "stale_state_refresh"}
    assert {
        row["difficulty"]["ambiguity_class"] for row in rows
    } == {"same_intent_command", "same_file_read", "stage_transition"}
    assert "external_trace_v3_semantic_required" not in json.dumps(rows)


def test_phase2m_head_dataset_uses_runtime_evidence_without_gold_prompt(
    tmp_path: Path,
) -> None:
    train_rows = normalize_phase2m_rows(
        build_phase2m_synthetic_safe_rows(split="train", count=24, repo_prefix="train"),
        split="train",
        collection_script_hash=_hash("phase2m-test-generator"),
    )
    val_rows = normalize_phase2m_rows(
        build_phase2m_synthetic_safe_rows(split="val", count=24, repo_prefix="val"),
        split="val",
        collection_script_hash=_hash("phase2m-test-generator"),
    )
    train = _write_jsonl(tmp_path / "train.jsonl", train_rows)
    val = _write_jsonl(tmp_path / "val.jsonl", val_rows)

    manifest = build_phase2m_head_dataset(
        train_jsonl=train,
        val_jsonl=val,
        output_dir=tmp_path / "heads",
    )
    head_row = json.loads(
        (tmp_path / "heads" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )

    assert manifest["sealed_v3_used"] is False
    assert manifest["json_text_target"] is False
    assert head_row["action_type"] == "RUN_COMMAND"
    assert head_row["head_scope"] == "debug_cortex"
    assert head_row["command"] in head_row["candidate_commands"]
    assert "expected_command" not in head_row["state_prompt"]
    assert "gold" not in head_row["state_prompt"].lower()


def test_phase2m_trace_to_head_row_rejects_missing_expected_candidate() -> None:
    row = build_phase2m_synthetic_safe_rows(
        split="val",
        count=1,
        repo_prefix="val",
    )[0]
    row["expected_command"] = "python -m pytest tests/missing.py::test_missing"

    try:
        phase2m_trace_to_head_row(row)
    except ValueError as exc:
        assert "expected_command not present" in str(exc)
    else:
        raise AssertionError("missing expected command should be rejected")


def test_phase2m_design_maturity_blocks_synthetic_declared_baseline_claim_training(
    tmp_path: Path,
) -> None:
    train = _write_jsonl(
        tmp_path / "train.jsonl",
        normalize_phase2m_rows(
            build_phase2m_synthetic_safe_rows(
                split="train", count=24, repo_prefix="train"
            ),
            split="train",
            collection_script_hash=_hash("phase2m-test-generator"),
        ),
    )
    val = _write_jsonl(
        tmp_path / "val.jsonl",
        normalize_phase2m_rows(
            build_phase2m_synthetic_safe_rows(split="val", count=24, repo_prefix="val"),
            split="val",
            collection_script_hash=_hash("phase2m-test-generator"),
        ),
    )
    holdout = _write_jsonl(
        tmp_path / "holdout.jsonl",
        normalize_phase2m_rows(
            build_phase2m_synthetic_safe_rows(
                split="holdout", count=12, repo_prefix="holdout"
            ),
            split="holdout",
            collection_script_hash=_hash("phase2m-test-generator"),
        ),
    )
    data_health = _write(
        tmp_path / "data_health.json",
        build_phase2m_data_health(
            train_jsonl=train,
            val_jsonl=val,
            holdout_jsonl=holdout,
        ),
    )
    head_manifest = _write(
        tmp_path / "head_manifest.json",
        build_phase2m_head_dataset(
            train_jsonl=train,
            val_jsonl=val,
            output_dir=tmp_path / "heads",
        ),
    )

    report = build_phase2m_design_maturity_review(
        data_health_json=data_health,
        head_manifest_json=head_manifest,
    )

    assert report["ready_for_plumbing_smoke"] is True
    assert report["ready_for_claim_bearing_training"] is False
    assert "do_not_treat_phase2m_smoke_as_claim_bearing_evidence" in report[
        "blocked_actions"
    ]
    assert report["checks"]["has_claim_bearing_public_repo_trace"] is False
    assert report["checks"]["baselines_are_measured_not_declared_only"] is False
    assert report["checks"]["runtime_evidence_avoids_direct_candidate_slot_marker"] is False


def test_phase2m_design_maturity_accepts_public_measured_trace_design(
    tmp_path: Path,
) -> None:
    train_rows = _phase2m_rows("train", repo_prefix="public_train", count=24)
    val_rows = _phase2m_rows("val", repo_prefix="public_val", count=24)
    holdout_rows = _phase2m_rows("holdout", repo_prefix="public_holdout", count=12)
    for split_rows in (train_rows, val_rows, holdout_rows):
        for index, row in enumerate(split_rows):
            if index % 4 == 0:
                row["baselines"]["native_head_only"] = row["expected_command"]
            row["source_kind"] = "public_repo"
            row["runtime_visible_evidence"] = {
                "traceback_symbols": ["billing.rounding_error"],
                "changed_files": ["src/billing/rounding.py"],
                "watched_files": ["tests/billing/test_rounding.py"],
                "module_owner": "billing",
            }
            row["baseline_metadata"] = {
                baseline: {"measured": True, "method": f"{baseline}_evaluator_v1"}
                for baseline in (
                    "source_overlap",
                    "native_head_only",
                    "continuation_only",
                    "prompt_only",
                    "react",
                )
            }
    for row in train_rows + val_rows + holdout_rows:
        row["source_kind"] = "public_repo"
    train = _write_jsonl(tmp_path / "train.jsonl", train_rows)
    val = _write_jsonl(tmp_path / "val.jsonl", val_rows)
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", holdout_rows)
    data_health_payload = build_phase2m_data_health(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
    )
    data_health = _write(tmp_path / "data_health.json", data_health_payload)
    head_manifest = _write(
        tmp_path / "head_manifest.json",
        build_phase2m_head_dataset(
            train_jsonl=train,
            val_jsonl=val,
            output_dir=tmp_path / "heads",
        ),
    )

    report = build_phase2m_design_maturity_review(
        data_health_json=data_health,
        head_manifest_json=head_manifest,
    )

    assert report["ready_for_claim_bearing_training"] is True


def test_phase2m_design_maturity_blocks_head_state_source_overlap_solved_design(
    tmp_path: Path,
) -> None:
    train_rows = _phase2m_rows("train", repo_prefix="public_train", count=24)
    val_rows = _phase2m_rows("val", repo_prefix="public_val", count=24)
    holdout_rows = _phase2m_rows("holdout", repo_prefix="public_holdout", count=12)
    for row in train_rows + val_rows + holdout_rows:
        row["source_kind"] = "public_repo"
        row["baseline_metadata"] = {
            baseline: {"measured": True, "method": f"{baseline}_evaluator_v1"}
            for baseline in (
                "source_overlap",
                "native_head_only",
                "continuation_only",
                "prompt_only",
                "react",
            )
        }
    for row in val_rows:
        row["runtime_visible_evidence"]["shortcut_visible_text"] = row["expected_command"]
    train = _write_jsonl(tmp_path / "train.jsonl", train_rows)
    val = _write_jsonl(tmp_path / "val.jsonl", val_rows)
    holdout = _write_jsonl(tmp_path / "holdout.jsonl", holdout_rows)
    data_health = _write(
        tmp_path / "data_health.json",
        build_phase2m_data_health(
            train_jsonl=train,
            val_jsonl=val,
            holdout_jsonl=holdout,
        ),
    )
    head_manifest = _write(
        tmp_path / "head_manifest.json",
        build_phase2m_head_dataset(
            train_jsonl=train,
            val_jsonl=val,
            output_dir=tmp_path / "heads",
        ),
    )
    head_state_baseline = _write(
        tmp_path / "head_state_baseline.json",
        build_phase2m_head_state_baseline_audit(
            head_train_jsonl=tmp_path / "heads" / "train.jsonl",
            head_val_jsonl=tmp_path / "heads" / "val.jsonl",
            max_val_source_overlap_accuracy=0.30,
        ),
    )

    report = build_phase2m_design_maturity_review(
        data_health_json=data_health,
        head_manifest_json=head_manifest,
        head_state_baseline_json=head_state_baseline,
    )

    assert report["ready_for_claim_bearing_training"] is False
    assert report["checks"]["head_state_source_overlap_pressure_not_solved"] is False
    assert "revise_phase2m_head_prompt_or_trace_design_before_training" in report[
        "blocked_actions"
    ]


def test_phase2m_v2_boundary_keeps_synthetic_split_as_plumbing_only() -> None:
    report = build_phase2m_v2_boundary()

    assert report["phase2m_v1_boundary"]["claim_bearing_training_allowed"] is False
    assert report["sealed_policy"]["sealed_v3_is_final_eval_only"] is True
    assert report["monitoring_policy"]["codex_automation_allowed"] is False


def test_phase2m_v2_strict_synthetic_rows_remove_candidate_markers_and_measure_baselines(
    tmp_path: Path,
) -> None:
    rows = build_phase2m_v2_synthetic_safe_rows(
        split="val",
        count=24,
        repo_prefix="val",
    )

    serialized = json.dumps(rows)
    assert "candidate_0" not in serialized
    assert "candidate-0" not in serialized
    for row in rows:
        assert row["baselines"] == compute_phase2m_baseline_predictions(row)
        assert all(payload["measured"] for payload in row["baseline_metadata"].values())

    train = _write_jsonl(
        tmp_path / "train.jsonl",
        normalize_phase2m_rows(
            build_phase2m_v2_synthetic_safe_rows(
                split="train", count=24, repo_prefix="train"
            ),
            split="train",
            collection_script_hash=_hash("phase2m-v2-test-generator"),
        ),
    )
    val = _write_jsonl(
        tmp_path / "val.jsonl",
        normalize_phase2m_rows(
            rows,
            split="val",
            collection_script_hash=_hash("phase2m-v2-test-generator"),
        ),
    )
    holdout = _write_jsonl(
        tmp_path / "holdout.jsonl",
        normalize_phase2m_rows(
            build_phase2m_v2_synthetic_safe_rows(
                split="holdout", count=12, repo_prefix="holdout"
            ),
            split="holdout",
            collection_script_hash=_hash("phase2m-v2-test-generator"),
        ),
    )

    report = build_phase2m_data_health(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
        require_measured_baselines=True,
        require_computed_baselines_match=True,
        forbid_candidate_slot_markers=True,
        require_all_baselines_below_threshold=True,
    )

    assert report["passed"] is True
    assert report["checks"]["phase2m_baseline_metadata_measured"] is True
    assert report["checks"]["phase2m_baselines_match_computed_predictions"] is True
    assert report["checks"]["phase2m_no_candidate_slot_marker_visible"] is True
    assert report["checks"]["phase2m_all_required_baselines_val_below_threshold"] is True


def test_phase2m_v2_data_health_rejects_candidate_marker_and_stale_baseline(
    tmp_path: Path,
) -> None:
    rows = normalize_phase2m_rows(
        build_phase2m_v2_synthetic_safe_rows(split="val", count=24, repo_prefix="val"),
        split="val",
        collection_script_hash=_hash("phase2m-v2-test-generator"),
    )
    rows[0]["current_visible_text"] += " candidate_0"
    rows[1]["baselines"]["source_overlap"] = rows[1]["expected_command"]
    train = _write_jsonl(
        tmp_path / "train.jsonl",
        normalize_phase2m_rows(
            build_phase2m_v2_synthetic_safe_rows(
                split="train", count=24, repo_prefix="train"
            ),
            split="train",
            collection_script_hash=_hash("phase2m-v2-test-generator"),
        ),
    )
    val = _write_jsonl(tmp_path / "val.jsonl", rows)
    holdout = _write_jsonl(
        tmp_path / "holdout.jsonl",
        normalize_phase2m_rows(
            build_phase2m_v2_synthetic_safe_rows(
                split="holdout", count=12, repo_prefix="holdout"
            ),
            split="holdout",
            collection_script_hash=_hash("phase2m-v2-test-generator"),
        ),
    )

    report = build_phase2m_data_health(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
        require_measured_baselines=True,
        require_computed_baselines_match=True,
        forbid_candidate_slot_markers=True,
        require_all_baselines_below_threshold=True,
    )

    assert report["passed"] is False
    assert "do_not_train_with_candidate_slot_markers_visible" in report["blocked_actions"]
    assert "do_not_train_with_declared_or_stale_phase2m_baselines" in report[
        "blocked_actions"
    ]


def test_phase2m_v2_postflight_blocks_when_design_maturity_is_not_claim_bearing(
    tmp_path: Path,
) -> None:
    summary = _write(
        tmp_path / "summary.json",
        {
            "train_examples": 128,
            "val_examples": 24,
            "use_pairwise_command_reranker": False,
            "no_json_motor_target": True,
            "low_level_qwen_calls_target": 0,
            "effective_split_hashes": {
                "phase2c_head_train": "train-hash",
                "phase2c_head_val": "val-hash",
            },
            "source_overlap_command_slot_baseline": {
                "val": {"total": 24, "accuracy": 0.20}
            },
            "config": {
                "max_train_records": 128,
                "max_val_records": 24,
                "latent_fusion": "additive",
            },
            "head_config": {
                "latent_fusion": "additive",
                "use_pairwise_command_reranker": False,
            },
            "history": [
                {
                    "train_pairwise_encoded_candidates": 0,
                    "val_metrics": {
                        "command_slot_accuracy": 0.90,
                        "command_slot_count": 24,
                        "pairwise_encoded_candidates": 0,
                    },
                }
            ],
            "run_manifest": {"finished_at_utc": "2026-05-21T00:00:00Z"},
        },
    )
    data_health = _write(tmp_path / "data.json", {"passed": True})
    pretrain = _write(tmp_path / "pretrain.json", {"passed": True})
    design = _write(
        tmp_path / "design.json",
        {"passed": False, "ready_for_claim_bearing_training": False},
    )
    head_manifest = _write(
        tmp_path / "manifest.json",
        {
            "splits": {
                "train": {"sha256": "train-hash"},
                "val": {"sha256": "val-hash"},
            }
        },
    )

    report = build_phase2m_v2_postflight(
        training_summary_json=summary,
        data_health_json=data_health,
        pretrain_gate_json=pretrain,
        design_maturity_json=design,
        head_manifest_json=head_manifest,
    )

    assert report["passed"] is False
    assert "do_not_treat_phase2m_training_as_claim_bearing_evidence" in report[
        "blocked_actions"
    ]


def test_phase2m_v2_postflight_allows_full_after_smoke_delta_gate(tmp_path: Path) -> None:
    summary = _write(
        tmp_path / "summary.json",
        {
            "train_examples": 128,
            "val_examples": 24,
            "use_pairwise_command_reranker": False,
            "no_json_motor_target": True,
            "low_level_qwen_calls_target": 0,
            "effective_split_hashes": {
                "phase2c_head_train": "train-hash",
                "phase2c_head_val": "val-hash",
            },
            "source_overlap_command_slot_baseline": {
                "val": {"total": 24, "accuracy": 0.20}
            },
            "config": {
                "max_train_records": 128,
                "max_val_records": 24,
                "latent_fusion": "additive",
            },
            "head_config": {
                "latent_fusion": "additive",
                "use_pairwise_command_reranker": False,
            },
            "history": [
                {
                    "train_pairwise_encoded_candidates": 0,
                    "val_metrics": {
                        "command_slot_accuracy": 0.90,
                        "command_slot_count": 24,
                        "pairwise_encoded_candidates": 0,
                    },
                }
            ],
            "run_manifest": {"finished_at_utc": "2026-05-21T00:00:00Z"},
        },
    )
    data_health = _write(tmp_path / "data.json", {"passed": True})
    pretrain = _write(tmp_path / "pretrain.json", {"passed": True})
    design = _write(
        tmp_path / "design.json",
        {"passed": True, "ready_for_claim_bearing_training": True},
    )
    head_manifest = _write(
        tmp_path / "manifest.json",
        {
            "splits": {
                "train": {"sha256": "train-hash"},
                "val": {"sha256": "val-hash"},
            }
        },
    )

    report = build_phase2m_v2_postflight(
        training_summary_json=summary,
        data_health_json=data_health,
        pretrain_gate_json=pretrain,
        design_maturity_json=design,
        head_manifest_json=head_manifest,
    )

    assert report["passed"] is True
    assert report["ready_for_full_train"] is True


def test_phase2m_v2_full_postflight_uses_command_slot_accuracy_as_completion(
    tmp_path: Path,
) -> None:
    summary = _write(
        tmp_path / "summary.json",
        {
            "train_examples": 512,
            "val_examples": 128,
            "use_pairwise_command_reranker": False,
            "no_json_motor_target": True,
            "low_level_qwen_calls_target": 0,
            "effective_split_hashes": {
                "phase2c_head_train": "train-hash",
                "phase2c_head_val": "val-hash",
            },
            "source_overlap_command_slot_baseline": {
                "val": {"total": 128, "accuracy": 0.40}
            },
            "config": {
                "max_train_records": 1024,
                "max_val_records": 512,
                "latent_fusion": "additive",
            },
            "head_config": {
                "latent_fusion": "additive",
                "use_pairwise_command_reranker": False,
            },
            "history": [
                {
                    "train_pairwise_encoded_candidates": 0,
                    "val_metrics": {
                        "command_slot_accuracy": 0.92,
                        "command_slot_count": 128,
                        "pairwise_encoded_candidates": 0,
                    },
                }
            ],
            "run_manifest": {"finished_at_utc": "2026-05-21T00:00:00Z"},
        },
    )
    data_health = _write(tmp_path / "data.json", {"passed": True})
    pretrain = _write(tmp_path / "pretrain.json", {"passed": True})
    design = _write(
        tmp_path / "design.json",
        {"passed": True, "ready_for_claim_bearing_training": True},
    )
    native = _write(tmp_path / "native.json", {"metrics": {"task_completion_rate": 0.70}})
    head_manifest = _write(
        tmp_path / "manifest.json",
        {
            "splits": {
                "train": {"sha256": "train-hash"},
                "val": {"sha256": "val-hash"},
            }
        },
    )

    report = build_phase2m_v2_postflight(
        training_summary_json=summary,
        data_health_json=data_health,
        pretrain_gate_json=pretrain,
        design_maturity_json=design,
        head_manifest_json=head_manifest,
        native_head_only_eval_json=native,
        postflight_stage="full",
    )

    assert report["passed"] is True
    assert report["ready_for_package"] is True
    assert report["metrics"]["full_completion"] == 0.92
    assert report["metrics"]["full_completion_source"] == "val_command_slot_accuracy"


def test_phase2m_public_repo_collector_emits_readonly_measured_traces(
    tmp_path: Path,
) -> None:
    train_repo = _make_public_pytest_repo(tmp_path, "train_repo")
    val_repo = _make_public_pytest_repo(tmp_path, "val_repo")
    holdout_repo = _make_public_pytest_repo(tmp_path, "holdout_repo")
    specs = [
        {"repo_id": "train_repo", "split": "train", "local_path": str(train_repo)},
        {"repo_id": "val_repo", "split": "val", "local_path": str(val_repo)},
        {"repo_id": "holdout_repo", "split": "holdout", "local_path": str(holdout_repo)},
    ]

    manifest = collect_phase2m_public_repo_traces(
        repo_specs=specs,
        clone_root=tmp_path / "clones",
        output_root=tmp_path / "phase2m_public",
        rows_per_repo=24,
        no_clone=True,
    )

    assert manifest["sealed_v3_used"] is False
    assert manifest["writes_to_collected_repos"] is False
    assert manifest["splits"]["train"]["rows"] == 24
    val_rows = [
        json.loads(line)
        for line in (tmp_path / "phase2m_public" / "val.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert {row["source_kind"] for row in val_rows} == {"public_repo"}
    assert all(row["baselines"] == compute_phase2m_baseline_predictions(row) for row in val_rows)
    assert all(
        payload["measured"] is True
        for row in val_rows
        for payload in row["baseline_metadata"].values()
    )
    assert "candidate_0" not in json.dumps(val_rows)
    assert "external_trace_v3_semantic_required" not in json.dumps(val_rows)


def test_phase2m_public_repo_collector_feeds_claim_bearing_design_gate(
    tmp_path: Path,
) -> None:
    train_repo = _make_public_pytest_repo(tmp_path, "public_train")
    val_repo = _make_public_pytest_repo(tmp_path, "public_val")
    holdout_repo = _make_public_pytest_repo(tmp_path, "public_holdout")
    collect_phase2m_public_repo_traces(
        repo_specs=[
            {"repo_id": "public_train", "split": "train", "local_path": str(train_repo)},
            {"repo_id": "public_val", "split": "val", "local_path": str(val_repo)},
            {
                "repo_id": "public_holdout",
                "split": "holdout",
                "local_path": str(holdout_repo),
            },
        ],
        clone_root=tmp_path / "clones",
        output_root=tmp_path / "phase2m_public",
        rows_per_repo=24,
        no_clone=True,
    )
    train = tmp_path / "phase2m_public" / "train.jsonl"
    val = tmp_path / "phase2m_public" / "val.jsonl"
    holdout = tmp_path / "phase2m_public" / "holdout.jsonl"
    data_health_payload = build_phase2m_data_health(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
        require_measured_baselines=True,
        require_computed_baselines_match=True,
        forbid_candidate_slot_markers=True,
        require_all_baselines_below_threshold=True,
        max_command_slot_share=0.45,
    )
    data_health = _write(tmp_path / "data_health.json", data_health_payload)
    head_manifest = _write(
        tmp_path / "head_manifest.json",
        build_phase2m_head_dataset(
            train_jsonl=train,
            val_jsonl=val,
            output_dir=tmp_path / "heads",
        ),
    )

    report = build_phase2m_design_maturity_review(
        data_health_json=data_health,
        head_manifest_json=head_manifest,
    )

    assert data_health_payload["passed"] is True
    assert data_health_payload["checks"]["phase2m_command_slot_share_below_threshold"] is True
    assert report["ready_for_claim_bearing_training"] is True
    assert report["checks"]["has_claim_bearing_public_repo_trace"] is True


def test_phase2m_public_repo_collector_structured_watch_keys_are_source_overlap_hard(
    tmp_path: Path,
) -> None:
    train_repo = _make_public_pytest_repo(tmp_path, "watch_train")
    val_repo = _make_public_pytest_repo(tmp_path, "watch_val")
    holdout_repo = _make_public_pytest_repo(tmp_path, "watch_holdout")
    collect_phase2m_public_repo_traces(
        repo_specs=[
            {"repo_id": "watch_train", "split": "train", "local_path": str(train_repo)},
            {"repo_id": "watch_val", "split": "val", "local_path": str(val_repo)},
            {"repo_id": "watch_holdout", "split": "holdout", "local_path": str(holdout_repo)},
        ],
        clone_root=tmp_path / "clones",
        output_root=tmp_path / "phase2m_public",
        rows_per_repo=24,
        no_clone=True,
        structured_watch_keys=True,
        include_behavior_summary=False,
    )
    train = tmp_path / "phase2m_public" / "train.jsonl"
    val = tmp_path / "phase2m_public" / "val.jsonl"
    holdout = tmp_path / "phase2m_public" / "holdout.jsonl"
    data_health_payload = build_phase2m_data_health(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
        require_measured_baselines=True,
        require_computed_baselines_match=True,
        forbid_candidate_slot_markers=True,
        require_all_baselines_below_threshold=True,
        max_command_slot_share=0.45,
    )
    data_health = _write(tmp_path / "data_health.json", data_health_payload)
    head_manifest = _write(
        tmp_path / "head_manifest.json",
        build_phase2m_head_dataset(
            train_jsonl=train,
            val_jsonl=val,
            output_dir=tmp_path / "heads",
        ),
    )
    head_state_baseline = _write(
        tmp_path / "head_state_baseline.json",
        build_phase2m_head_state_baseline_audit(
            head_train_jsonl=tmp_path / "heads" / "train.jsonl",
            head_val_jsonl=tmp_path / "heads" / "val.jsonl",
            max_val_source_overlap_accuracy=0.50,
        ),
    )
    head_row = json.loads(
        (tmp_path / "heads" / "val.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    report = build_phase2m_design_maturity_review(
        data_health_json=data_health,
        head_manifest_json=head_manifest,
        head_state_baseline_json=head_state_baseline,
    )

    assert data_health_payload["passed"] is True
    assert report["ready_for_claim_bearing_training"] is True
    assert report["checks"]["head_state_source_overlap_pressure_not_solved"] is True
    assert head_row["nsi_reference"]["command_identity_confidence"] > 0.0
    assert "candidate_0" not in head_row["state_prompt"]


def test_phase2m_public_repo_collector_records_insufficient_repos(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "tiny_repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_tiny.py").write_text(
        "def test_one():\n    assert True\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "phase2m@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Phase2M Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "tiny fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    manifest = collect_phase2m_public_repo_traces(
        repo_specs=[{"repo_id": "tiny_repo", "split": "val", "local_path": str(repo)}],
        clone_root=tmp_path / "clones",
        output_root=tmp_path / "phase2m_public",
        rows_per_repo=4,
        no_clone=True,
    )

    assert manifest["splits"]["val"]["rows"] == 0
    assert "no_test_file_with_four_or_more_pytest_functions" in manifest["repos"][0][
        "rejected_reasons"
    ]
