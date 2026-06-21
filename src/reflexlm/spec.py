from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PACKAGE_ROOT / "configs" / "project_spec.yaml"


@lru_cache(maxsize=1)
def load_project_spec() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def action_space() -> list[str]:
    return list(load_project_spec()["action_space"])


def task_suite() -> list[str]:
    return list(load_project_spec()["task_suite"])


def dataset_target_episode_count() -> int:
    return int(load_project_spec()["dataset"]["target_episode_count"])

