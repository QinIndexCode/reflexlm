"""Computer-native sensory-motor core experiments.

This namespace is intentionally separate from the current NSI model used by
the bounded evidence pipeline. ReflexCore V0 reuses the existing receptor,
feature, and safety layers, but it does not replace them.
"""

from reflexlm.core.schema import (
    MOTOR_ACTIONS,
    ComputerObservation,
    MotorAction,
    ReflexCoreTrainingExample,
    action_from_index,
    action_to_index,
)
from reflexlm.core.model import ReflexCoreV0, ReflexCoreV0Config
from reflexlm.core.motor import ReflexCoreDecodedMotor, ReflexCoreMotorConfig, decode_reflexcore_motor
from reflexlm.core.observation import ReflexCoreObservationContext
from reflexlm.core.experience import (
    ReflexCoreExperienceSummary,
    examples_from_step_trace,
    summarize_experience,
    write_experience_jsonl,
)
from reflexlm.core.online_adaptation import (
    ReflexCoreOnlineAdaptationConfig,
    adapt_reflexcore_from_experience,
)
from reflexlm.core.online_adaptation_gate import (
    ReflexCoreFamilyHoldoutMatrixConfig,
    ReflexCoreOnlineAdaptationGateConfig,
    ReflexCoreOnlineAdaptationSplit,
    run_family_holdout_matrix,
    run_online_adaptation_gate,
    split_online_adaptation_examples,
)
from reflexlm.core.real_sandbox_capability_matrix import (
    ReflexCoreRealSandboxCapabilityMatrixConfig,
    run_reflexcore_real_sandbox_capability_matrix,
)

__all__ = [
    "MOTOR_ACTIONS",
    "ComputerObservation",
    "MotorAction",
    "ReflexCoreTrainingExample",
    "ReflexCoreV0",
    "ReflexCoreV0Config",
    "ReflexCoreDecodedMotor",
    "ReflexCoreExperienceSummary",
    "ReflexCoreFamilyHoldoutMatrixConfig",
    "ReflexCoreMotorConfig",
    "ReflexCoreObservationContext",
    "ReflexCoreOnlineAdaptationConfig",
    "ReflexCoreOnlineAdaptationGateConfig",
    "ReflexCoreOnlineAdaptationSplit",
    "ReflexCoreRealSandboxCapabilityMatrixConfig",
    "action_from_index",
    "action_to_index",
    "adapt_reflexcore_from_experience",
    "decode_reflexcore_motor",
    "examples_from_step_trace",
    "run_family_holdout_matrix",
    "run_online_adaptation_gate",
    "run_reflexcore_real_sandbox_capability_matrix",
    "split_online_adaptation_examples",
    "summarize_experience",
    "write_experience_jsonl",
]
