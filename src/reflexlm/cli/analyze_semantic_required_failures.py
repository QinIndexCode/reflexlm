from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return list(json.loads(text))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _load_eval(eval_json: Path) -> dict[str, Any]:
    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    run_path = Path(payload["run_path"])
    traces = _read_json_or_jsonl(run_path / "trace_rows.jsonl")
    episodes = _read_json_or_jsonl(run_path / "episode_results.jsonl")
    return {"payload": payload, "run_path": run_path, "traces": traces, "episodes": episodes}


def analyze_semantic_required_failures(
    *,
    metadata_json: str | Path,
    eval_jsons: list[str | Path],
    output_json: str | Path,
) -> dict[str, Any]:
    metadata_rows = _read_json_or_jsonl(Path(metadata_json))
    metadata = {row["episode_id"]: row for row in metadata_rows}
    policies: dict[str, Any] = {}
    for eval_path_value in eval_jsons:
        eval_path = Path(eval_path_value)
        loaded = _load_eval(eval_path)
        policy_label = str(loaded["payload"]["policy"]["policy_label"])
        traces_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trace in loaded["traces"]:
            traces_by_episode[str(trace["episode_id"])].append(trace)
        scenario_totals: Counter[str] = Counter()
        scenario_success: Counter[str] = Counter()
        scenario_predictions: dict[str, Counter[str]] = defaultdict(Counter)
        scenario_oracles: dict[str, Counter[str]] = defaultdict(Counter)
        for episode in loaded["episodes"]:
            episode_id = str(episode["episode_id"])
            scenario = str(metadata.get(episode_id, {}).get("scenario_template", "unknown"))
            scenario_totals[scenario] += 1
            if float(episode.get("task_completion_rate") or 0.0) >= 1.0:
                scenario_success[scenario] += 1
            final_trace = traces_by_episode[episode_id][-1]
            predicted = str(final_trace["action"].get("command") or final_trace["action"].get("type"))
            oracle = str(
                final_trace["oracle_action"].get("command")
                or final_trace["oracle_action"].get("type")
            )
            scenario_predictions[scenario][predicted] += 1
            scenario_oracles[scenario][oracle] += 1
        policies[policy_label] = {
            "eval_json": str(eval_path),
            "run_path": str(loaded["run_path"]),
            "completion": loaded["payload"]["metrics"]["aggregate"]["task_completion_rate"],
            "scenario_summary": {
                scenario: {
                    "success": int(scenario_success[scenario]),
                    "episodes": int(scenario_totals[scenario]),
                    "predicted_commands": dict(scenario_predictions[scenario]),
                    "oracle_commands": dict(scenario_oracles[scenario]),
                }
                for scenario in sorted(scenario_totals)
            },
        }
    report = {
        "analysis": "semantic_required_mechanism_failure_audit",
        "metadata_json": str(metadata_json),
        "sealed_data_used_for_training": False,
        "policies": policies,
    }
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit semantic-required external-trace failures without generating training data."
    )
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--eval-json", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = analyze_semantic_required_failures(
        metadata_json=args.metadata_json,
        eval_jsons=args.eval_json,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
