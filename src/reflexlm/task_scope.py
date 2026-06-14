from __future__ import annotations

from reflexlm.schema import TaskType


PHASE1_ALL_HARD_TASKS = frozenset(
    {
        TaskType.TEST_FAILURE,
        TaskType.FILE_CHANGE,
        TaskType.ROUTINE_RECOVERY,
    }
)

REFLEX_LAYER_TASKS = frozenset(
    {
        TaskType.BLOCKING_INPUT,
        TaskType.PROCESS_HANG,
        TaskType.DANGEROUS_ACTION,
        TaskType.FILE_CHANGE,
        TaskType.ROUTINE_RECOVERY,
    }
)

DEBUG_CORTEX_TASKS = frozenset({TaskType.TEST_FAILURE})

REFLEX_LAYER_HARD_TASKS = frozenset(
    {
        TaskType.PROCESS_HANG,
        TaskType.FILE_CHANGE,
        TaskType.ROUTINE_RECOVERY,
    }
)


def task_values(tasks: frozenset[TaskType]) -> set[str]:
    return {task.value for task in tasks}


def scope_for_task(task_type: TaskType | str) -> str:
    task = task_type if isinstance(task_type, TaskType) else TaskType(task_type)
    if task in DEBUG_CORTEX_TASKS:
        return "debug_cortex"
    if task in REFLEX_LAYER_TASKS:
        return "reflex_layer"
    return "unknown"


def tasks_for_scope(scope: str) -> frozenset[TaskType]:
    if scope == "all":
        return frozenset(TaskType)
    if scope == "reflex_layer":
        return REFLEX_LAYER_TASKS
    if scope == "debug_cortex":
        return DEBUG_CORTEX_TASKS
    raise ValueError(f"Unsupported task scope: {scope}")


def hard_tasks_for_set(name: str) -> frozenset[TaskType]:
    if name == "phase1_all":
        return PHASE1_ALL_HARD_TASKS
    if name == "reflex_layer":
        return REFLEX_LAYER_HARD_TASKS
    raise ValueError(f"Unsupported hard task set: {name}")
