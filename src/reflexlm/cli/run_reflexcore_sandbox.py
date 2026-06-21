from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from reflexlm.core.experience import write_experience_jsonl
from reflexlm.core.model import ReflexCoreV0, ReflexCoreV0Config
from reflexlm.core.runner import ReflexCoreSandboxConfig, ReflexCoreSandboxRunner
from reflexlm.models.features import StateVectorizer
from reflexlm.schema import ActionDecision, ActionType, GoalSpec, TaskType


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one bounded ReflexCore V0 sandbox step.",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--sandbox-root", required=True)
    parser.add_argument("--allowed-command", action="append", default=[])
    parser.add_argument("--description", default="Bounded ReflexCore sandbox smoke")
    parser.add_argument("--execute", action="store_true", help="Execute allowlisted commands in sandbox")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--loop", action="store_true", help="Run recurrent observe/propose/step loop")
    parser.add_argument(
        "--live-observation",
        action="store_true",
        help="Re-observe terminal/process/filesystem/time receptors after each action",
    )
    parser.add_argument(
        "--write-experience",
        help="Write post-safety model rollout transitions as ReflexCore JSONL examples",
    )
    parser.add_argument("--episode-id", default="reflexcore-sandbox-model-rollout")
    args = parser.parse_args()

    if args.checkpoint:
        checkpoint = torch.load(Path(args.checkpoint), map_location="cpu")
        config = ReflexCoreV0Config(**checkpoint["config"])
        model = ReflexCoreV0(config)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        input_dim = StateVectorizer().vector_dim
        model = ReflexCoreV0(ReflexCoreV0Config.smoke(input_dim=input_dim))
    runner = ReflexCoreSandboxRunner(
        ReflexCoreSandboxConfig(
            sandbox_root=Path(args.sandbox_root),
            allowed_commands=tuple(args.allowed_command),
            allow_process_execution=args.execute,
            max_steps=args.steps,
        )
    )
    goal = GoalSpec(
        task_type=TaskType.ROUTINE_RECOVERY,
        description=args.description,
        command_allowlist=list(args.allowed_command),
        watched_paths=[str(Path(args.sandbox_root))],
        success_criteria=["sandbox step remains inside allowlist"],
    )
    if args.loop:
        if args.live_observation:
            live_loop = runner.run_model_live_observation_loop(model, goal)
            state = live_loop.initial_state
            trace = live_loop.trace
        else:
            state = runner.initial_state(goal)
            trace = runner.run_model_loop(model, state)
        experience = (
            write_experience_jsonl(
                Path(args.write_experience),
                initial_state=state,
                trace=trace,
                episode_id=args.episode_id,
                vocab_size=model.config.vocab_size,
                max_text_tokens=128,
            )
            if args.write_experience
            else None
        )
        payload = {
            "live_observation": args.live_observation,
            "trace": [
                {
                    "allowed": item.safety_decision.allowed,
                    "reason": item.safety_decision.reason,
                    "action": (
                        item.safety_decision.action.model_dump(mode="json")
                        if item.safety_decision.action is not None
                        else None
                    ),
                    "stdout": item.stdout,
                    "stderr": item.stderr,
                    "done": item.done,
                    "tick": item.state.time.tick,
                    "changed_paths": item.state.filesystem.changed_paths,
                    "model_prediction_error": item.model_prediction_error,
                    "observed_prediction_error": item.observed_prediction_error,
                }
                for item in trace
            ],
            "experience": (
                asdict(experience)
                if experience is not None
                else None
            ),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.live_observation:
        context = runner.live_observation_context(
            goal,
            vocab_size=model.config.vocab_size,
        )
        state = context.observe_state(prompt_visible=True)
    else:
        context = None
        state = runner.initial_state(goal)
    proposal = runner.propose_with_state(model, state)
    safety = proposal.safety_decision
    action = safety.action or ActionDecision(
        type=ActionType.BLOCK,
        reason=safety.reason,
        confidence=1.0,
    )
    result = runner.step(state, action)
    result = runner.attach_prediction(result, proposal)
    if context is not None:
        result = runner.reobserve_step_result(context, result)
    experience = (
        write_experience_jsonl(
            Path(args.write_experience),
            initial_state=state,
            trace=[result],
            episode_id=args.episode_id,
            vocab_size=model.config.vocab_size,
            max_text_tokens=128,
        )
        if args.write_experience
        else None
    )
    payload = {
        "live_observation": args.live_observation,
        "allowed": safety.allowed,
        "reason": safety.reason,
        "action": safety.action.model_dump(mode="json") if safety.action else None,
        "result": {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "done": result.done,
            "tick": result.state.time.tick,
            "changed_paths": result.state.filesystem.changed_paths,
            "model_prediction_error": result.model_prediction_error,
            "observed_prediction_error": result.observed_prediction_error,
        },
        "experience": (
            asdict(experience)
            if experience is not None
            else None
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
