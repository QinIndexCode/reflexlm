from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reflexlm.baselines.text_policies import (
    HuggingFaceJSONPolicy,
    TextPolicyStats,
    scaled_generation_budget,
)
from reflexlm.eval import PolicyStats, SequenceModelPolicy
from reflexlm.llm.prompts import SynapseSummary
from reflexlm.models.features import candidate_commands, candidate_files
from reflexlm.schema import ActionDecision, ActionType, SystemStateFrame
from reflexlm.train import load_model_checkpoint


@dataclass(slots=True)
class HybridPolicyConfig:
    base_model_name: str
    shared_adapter_path: str | None = None
    adapter_map: dict[str, str] = field(default_factory=dict)
    quantization: str = "4bit"
    confidence_threshold: float = 0.72
    prediction_error_threshold: float = 0.45
    risk_threshold: float = 0.7
    max_new_tokens: int = 96
    max_time_s: float | None = 20.0
    max_retries: int = 1
    parse_retry_growth: float = 1.0
    cpu_offload: bool = False
    direct_action_types: tuple[str, ...] = (
        "WAIT",
        "STOP_PROCESS",
        "ASK_USER",
        "REFRESH_STATE",
        "BLOCK",
        "DONE",
    )
    prompt_style: str = "synapse_augmented"


class AdapterJSONPolicy(HuggingFaceJSONPolicy):
    def __init__(
        self,
        model_name: str,
        *,
        adapter_path: str | Path | None = None,
        adapter_map: dict[str, str] | None = None,
        prompt_style: str = "prompt_only",
        policy_label: str = "qwen_adapter",
        **kwargs: Any,
    ) -> None:
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.adapter_map = adapter_map or {}
        self.prompt_style = prompt_style
        super().__init__(
            model_name,
            react_style=prompt_style == "react",
            policy_label=policy_label,
            **kwargs,
        )
        if adapter_path or self.adapter_map:
            from peft import PeftModel

            if self.adapter_map and self.adapter_path is None:
                initial_route_name, initial_path = next(iter(self.adapter_map.items()))
                adapter_name = initial_route_name
            else:
                initial_path = self.adapter_path or next(iter(self.adapter_map.values()))
                adapter_name = Path(initial_path).name
            self.model = PeftModel.from_pretrained(self.model, initial_path, adapter_name=adapter_name)
            self.active_adapter_name = adapter_name
            for route_name, route_path in self.adapter_map.items():
                route_adapter_name = route_name
                if route_adapter_name == adapter_name:
                    continue
                self.model.load_adapter(route_path, adapter_name=route_adapter_name)
            self.loaded_adapters = list(self.adapter_map) or [adapter_name]
        else:
            self.active_adapter_name = None
            self.loaded_adapters = []

    def set_route_adapter(self, route_name: str | None) -> None:
        if route_name is None or not self.adapter_map:
            return
        if route_name not in self.adapter_map:
            return
        self.model.set_adapter(route_name)
        self.model.eval()
        self.active_adapter_name = route_name

    def metadata(self) -> dict[str, Any]:
        base = super().metadata()
        base.update(
            {
                "adapter_path": self.adapter_path,
                "adapter_map": self.adapter_map,
                "prompt_style": self.prompt_style,
            }
        )
        return base

    def act_with_synapse(
        self,
        state: SystemStateFrame,
        *,
        synapse_summary: SynapseSummary | None = None,
        route_name: str | None = None,
    ) -> ActionDecision:
        if route_name:
            self.set_route_adapter(route_name)
        return super().act(state) if synapse_summary is None else self._act_with_custom_prompt(state, synapse_summary)

    def _act_with_custom_prompt(
        self,
        state: SystemStateFrame,
        synapse_summary: SynapseSummary,
    ) -> ActionDecision:
        from reflexlm.llm.prompts import build_phase2_user_prompt, phase2_system_prompt

        user_prompt = build_phase2_user_prompt(
            state,
            prompt_style=self.prompt_style,
            synapse_summary=synapse_summary,
        )
        system_prompt = phase2_system_prompt(prompt_style=self.prompt_style)
        rendered_prompt = self._render_prompt(user_prompt) if not hasattr(self.tokenizer, "apply_chat_template") else self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
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
            outputs = self.model.generate(**inputs, **generate_kwargs)
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
                    "route_adapter": self.active_adapter_name,
                    "synapse_summary": synapse_summary.to_dict(),
                }
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
            "route_adapter": self.active_adapter_name,
            "synapse_summary": synapse_summary.to_dict(),
        }
        return ActionDecision(type=ActionType.WAIT, reason="llm_parse_failure", confidence=0.0)


class HybridSynapticPolicy:
    def __init__(
        self,
        *,
        nsi_checkpoint_path: str | Path,
        hybrid_config: HybridPolicyConfig,
        nsi_device: str = "cpu",
    ) -> None:
        model, vectorizer, _payload = load_model_checkpoint(nsi_checkpoint_path, device=nsi_device)
        self.nsi_policy = SequenceModelPolicy(model, vectorizer, policy_label="hybrid_nsi_gate")
        self.hybrid_config = hybrid_config
        self.llm_policy = AdapterJSONPolicy(
            hybrid_config.base_model_name,
            adapter_path=hybrid_config.shared_adapter_path,
            adapter_map=hybrid_config.adapter_map,
            prompt_style=hybrid_config.prompt_style,
            policy_label="hybrid_synaptic_qwen7b",
            quantization=hybrid_config.quantization,
            max_new_tokens=hybrid_config.max_new_tokens,
            max_time_s=hybrid_config.max_time_s,
            max_retries=hybrid_config.max_retries,
            parse_retry_growth=hybrid_config.parse_retry_growth,
            cpu_offload=hybrid_config.cpu_offload,
        )
        self.stats = PolicyStats()
        self.last_call: dict[str, Any] = {}

    def reset(self) -> None:
        self.nsi_policy.reset()
        self.llm_policy.reset()
        self.stats = PolicyStats()
        self.last_call = {}

    def metadata(self) -> dict[str, Any]:
        return {
            "policy_family": "hybrid_synaptic_qwen",
            "policy_label": "hybrid_synaptic_qwen7b",
            "hybrid_config": {
                "base_model_name": self.hybrid_config.base_model_name,
                "shared_adapter_path": self.hybrid_config.shared_adapter_path,
                "adapter_map": self.hybrid_config.adapter_map,
                "quantization": self.hybrid_config.quantization,
                "confidence_threshold": self.hybrid_config.confidence_threshold,
                "prediction_error_threshold": self.hybrid_config.prediction_error_threshold,
                "risk_threshold": self.hybrid_config.risk_threshold,
                "direct_action_types": list(self.hybrid_config.direct_action_types),
                "prompt_style": self.hybrid_config.prompt_style,
                "parse_retry_growth": self.hybrid_config.parse_retry_growth,
            },
        }

    def act(self, state: SystemStateFrame) -> ActionDecision:
        nsi_action = self.nsi_policy.act(state)
        nsi_debug = self.nsi_policy.last_call
        summary = SynapseSummary(
            route_name=str(nsi_debug["route_name"]),
            salience=float(nsi_debug["salience"]),
            risk=float(nsi_debug["risk"]),
            prediction_error=float(nsi_debug["prediction_error"]),
            confidence=float(nsi_debug["confidence"]),
            reflex_action=nsi_action.type.value,
            reflex_command=nsi_action.command,
            reflex_file_target=nsi_action.file_target,
        )
        if self._use_direct_reflex(nsi_action, summary):
            self.stats.token_cost = self.nsi_policy.stats.token_cost
            self.stats.model_calls = self.nsi_policy.stats.model_calls
            self.last_call = {"path": "nsi_direct", "nsi": nsi_debug}
            return nsi_action
        llm_action = self.llm_policy.act_with_synapse(
            state,
            synapse_summary=summary,
            route_name=summary.route_name if self.hybrid_config.adapter_map else None,
        )
        self.stats.token_cost = self.nsi_policy.stats.token_cost + self.llm_policy.stats.token_cost
        self.stats.model_calls = self.nsi_policy.stats.model_calls + self.llm_policy.stats.model_calls
        self.stats.parse_failures = self.llm_policy.stats.parse_failures
        self.stats.retries = self.llm_policy.stats.retries
        self.last_call = {
            "path": "llm_escalation",
            "nsi": nsi_debug,
            "llm": self.llm_policy.last_call,
        }
        return llm_action

    def _use_direct_reflex(self, action: ActionDecision, summary: SynapseSummary) -> bool:
        if action.type.value == "BLOCK":
            return True
        if action.type.value not in self.hybrid_config.direct_action_types:
            return False
        if summary.confidence < self.hybrid_config.confidence_threshold:
            return False
        if summary.prediction_error > self.hybrid_config.prediction_error_threshold:
            return False
        if summary.risk > self.hybrid_config.risk_threshold and action.type not in {
            ActionType.BLOCK,
            ActionType.ASK_USER,
        }:
            return False
        return True
