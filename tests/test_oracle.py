from reflexlm.data.tasks import TASK_VARIANTS, build_env
from reflexlm.runtime.oracle import RuleOracle
from reflexlm.schema import TaskType


def test_rule_oracle_completes_all_task_variants() -> None:
    oracle = RuleOracle()
    for task_type in TaskType:
        for index, _variant in enumerate(TASK_VARIANTS[task_type]):
            env = build_env(task_type, index)
            state = env.reset()
            done = False
            reward = -1.0
            steps = 0
            while not done and steps < env.max_steps:
                action = oracle.act(state)
                state, reward, done, _info = env.step(action)
                steps += 1
            assert done, f"{task_type.value} should terminate"
            assert reward > 0, f"{task_type.value} should complete successfully"

