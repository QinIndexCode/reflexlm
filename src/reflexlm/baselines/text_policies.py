from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from reflexlm.models.features import (
    ACTION_ORDER,
    candidate_commands,
    candidate_files,
    serialize_state_as_text,
    valid_action_mask,
)
from reflexlm.schema import ActionDecision, ActionType, SystemStateFrame


PROMPT_ONLY_TEMPLATE = """You are a Phase 1 reflex baseline.
Return only JSON with keys action, command, file_target.
Action space: WAIT, READ_STDOUT, READ_STDERR, READ_FILE, RUN_COMMAND, STOP_PROCESS, ASK_USER, REFRESH_STATE, BLOCK, DONE.
For RUN_COMMAND, copy one complete command_slot value exactly.
For READ_FILE, copy one complete file_slot value exactly.
For all other actions, command and file_target must be null.
Never invent a command, path, action, or capability.

Current state:
{state_text}
"""

REACT_TEMPLATE = """You are a ReAct baseline for a system-state task.
Think internally, then return only JSON with keys action, command, file_target.
Do not execute free-form shell commands.
For RUN_COMMAND, copy one complete command_slot value exactly.
For READ_FILE, copy one complete file_slot value exactly.
For all other actions, command and file_target must be null.
Never invent a command, path, action, or capability.

State:
{state_text}
"""


@dataclass(slots=True)
class TextPolicyStats:
    token_cost: int = 0
    model_calls: int = 0
    parse_failures: int = 0
    retries: int = 0


def scaled_generation_budget(
    *,
    max_new_tokens: int,
    max_time_s: float | None,
    attempt: int,
    parse_retry_growth: float,
) -> dict[str, int | float]:
    growth = max(float(parse_retry_growth), 1.0) ** max(attempt, 0)
    budget: dict[str, int | float] = {
        "max_new_tokens": max(1, int(round(max_new_tokens * growth))),
    }
    if max_time_s is not None:
        budget["max_time"] = max_time_s * growth
    return budget


class _MissingTransformers(RuntimeError):
    pass


class HuggingFaceJSONPolicy:
    def __init__(
        self,
        model_name: str,
        *,
        react_style: bool,
        quantization: str = "none",
        max_new_tokens: int = 96,
        max_time_s: float | None = 20.0,
        max_retries: int = 1,
        parse_retry_growth: float = 1.0,
        cpu_offload: bool = False,
        policy_label: str | None = None,
        maintain_history: bool = False,
        max_history_steps: int = 8,
    ) -> None:
        try:
            import torch
            from transformers import (  # type: ignore
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise _MissingTransformers(
                "transformers is required for HuggingFaceJSONPolicy; install the llm extras"
            ) from exc

        self.model_name = model_name
        self.react_style = react_style
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.max_time_s = max_time_s
        self.max_retries = max_retries
        self.parse_retry_growth = parse_retry_growth
        self.cpu_offload = cpu_offload
        self.maintain_history = maintain_history
        self.max_history_steps = max(1, max_history_steps)
        self.policy_label = policy_label or (
            "qwen_react_7b" if react_style else "qwen_prompt_only_7b"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        quantization_config = None
        if quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=cpu_offload,
            )
        elif quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
        elif quantization != "none":
            raise ValueError(f"Unsupported quantization mode: {quantization}")
        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "torch_dtype": "auto",
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.stats = TextPolicyStats()
        self.last_call: dict[str, Any] = {}
        self.history: list[dict[str, str]] = []

    def reset(self) -> None:
        self.stats = TextPolicyStats()
        self.last_call = {}
        self.history = []

    def metadata(self) -> dict[str, Any]:
        return {
            "policy_family": "huggingface_json",
            "policy_label": self.policy_label,
            "model_name": self.model_name,
            "react_style": self.react_style,
            "quantization": self.quantization,
            "max_new_tokens": self.max_new_tokens,
            "max_time_s": self.max_time_s,
            "max_retries": self.max_retries,
            "parse_retry_growth": self.parse_retry_growth,
            "cpu_offload": self.cpu_offload,
            "maintain_history": self.maintain_history,
            "max_history_steps": self.max_history_steps,
        }

    def act(self, state: SystemStateFrame) -> ActionDecision:  # pragma: no cover - optional path
        state_text = serialize_state_as_text(state, include_internal_hints=False)
        prompt = (REACT_TEMPLATE if self.react_style else PROMPT_ONLY_TEMPLATE).format(state_text=state_text)
        if self.maintain_history and self.history:
            prompt = (
                "Previous bounded tool-decision history:\n"
                f"{self._history_text()}\n\n"
                f"{prompt}"
            )
        rendered_prompt = self._render_prompt(prompt)
        parse_error = ""
        response = ""
        for attempt in range(self.max_retries + 1):
            inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
            input_device = next(self.model.parameters()).device
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
            generate_kwargs: dict[str, Any] = {
                "do_sample": False,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            generate_kwargs.update(
                scaled_generation_budget(
                    max_new_tokens=self.max_new_tokens,
                    max_time_s=self.max_time_s,
                    attempt=attempt,
                    parse_retry_growth=self.parse_retry_growth,
                )
            )
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
            generated = outputs[0][inputs["input_ids"].shape[-1] :]
            response = self.tokenizer.decode(generated, skip_special_tokens=True)
            self.stats.model_calls += 1
            self.stats.token_cost += int(inputs["input_ids"].shape[-1] + generated.shape[-1])
            try:
                action = self._parse_action(response)
                action = self._validate_or_fallback_action(action, state)
                self.last_call = {
                    "response_text": response,
                    "attempt": attempt + 1,
                    "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                    "generated_tokens": int(generated.shape[-1]),
                    "history_steps": len(self.history),
                }
                self._append_history(
                    state_text=state_text,
                    response=response,
                    action=action,
                )
                if attempt:
                    self.stats.retries += attempt
                return action
            except Exception as exc:
                parse_error = str(exc)
                self.stats.parse_failures += 1
                if attempt == self.max_retries:
                    break
        self.stats.retries += self.max_retries
        self.last_call = {
            "response_text": response,
            "parse_error": parse_error,
            "attempt": self.max_retries + 1,
            "history_steps": len(self.history),
        }
        fallback = ActionDecision(type=ActionType.WAIT, reason="llm_parse_failure", confidence=0.0)
        self._append_history(state_text=state_text, response=response, action=fallback)
        return fallback

    def _history_text(self) -> str:
        rows = self.history[-self.max_history_steps :]
        return "\n\n".join(
            f"Step {index + 1} state:\n{row['state']}\n"
            f"Step {index + 1} model response:\n{row['response']}\n"
            f"Step {index + 1} executed decision:\n{row['action']}"
            for index, row in enumerate(rows)
        )

    def _append_history(
        self,
        *,
        state_text: str,
        response: str,
        action: ActionDecision,
    ) -> None:
        if not self.maintain_history:
            return
        self.history.append(
            {
                "state": state_text,
                "response": response,
                "action": json.dumps(action.model_dump(mode="json"), ensure_ascii=False),
            }
        )
        if len(self.history) > self.max_history_steps:
            self.history = self.history[-self.max_history_steps :]

    def _render_prompt(self, user_prompt: str) -> str:
        system_prompt = (
            "You are a bounded Phase 1 system-state policy. "
            "Choose exactly one safe action from the fixed action space and return only JSON."
        )
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{system_prompt}\n\n{user_prompt}\n"

    def _parse_action(self, text: str) -> ActionDecision:
        for candidate in self._json_candidates(text):
            payload = json.loads(candidate)
            action_name = self._normalize_action_name(payload)
            return ActionDecision(
                type=ActionType(action_name),
                command=payload.get("command"),
                file_target=payload.get("file_target"),
                confidence=1.0,
            )
        raise ValueError("no valid JSON action object found in model response")

    def _normalize_action_name(self, payload: dict[str, Any]) -> str:
        raw = str(payload["action"]).strip().upper()
        if raw in {action.value for action in ActionType}:
            return raw
        compact = raw.replace("-", "_").replace(" ", "_")
        command = payload.get("command")
        file_target = payload.get("file_target")
        if command and any(
            token in compact
            for token in ("RUN", "RERUN", "TEST", "COMMAND", "REPAIR", "REACT")
        ):
            return ActionType.RUN_COMMAND.value
        if file_target and any(
            token in compact
            for token in ("FILE", "SOURCE", "OPEN", "READ", "INSPECT", "REACT")
        ):
            return ActionType.READ_FILE.value
        if any(token in compact for token in ("STDERR", "ERROR", "FAILURE", "TRACEBACK")):
            return ActionType.READ_STDERR.value
        if any(token in compact for token in ("STDOUT", "OUTPUT", "LOG")):
            return ActionType.READ_STDOUT.value
        if any(token in compact for token in ("WAIT", "SLEEP", "POLL")):
            return ActionType.WAIT.value
        if any(token in compact for token in ("DONE", "FINISH", "COMPLETE")):
            return ActionType.DONE.value
        return raw

    def _json_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []
        start = None
        depth = 0
        for index, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}":
                if depth:
                    depth -= 1
                    if depth == 0 and start is not None:
                        candidates.append(text[start : index + 1])
                start = None
        return candidates

    def _validate_or_fallback_action(
        self,
        action: ActionDecision,
        state: SystemStateFrame,
    ) -> ActionDecision:
        mask = valid_action_mask(state)
        action_index = ACTION_ORDER.index(action.type)
        if mask[action_index] <= 0.0:
            return ActionDecision(type=ActionType.WAIT, reason="llm_illegal_action_mask", confidence=0.0)
        if action.type == ActionType.RUN_COMMAND and action.command not in candidate_commands(state):
            return ActionDecision(type=ActionType.WAIT, reason="llm_invalid_command_candidate", confidence=0.0)
        if action.type == ActionType.READ_FILE and action.file_target not in candidate_files(state):
            return ActionDecision(type=ActionType.WAIT, reason="llm_invalid_file_candidate", confidence=0.0)
        return action
