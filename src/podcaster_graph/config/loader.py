from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from podcaster_graph.state import PodcasterState

# Reuse Crew definitions so agents/tasks stay single-sourced.
_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "podcaster" / "config"


def _load_yaml(name: str) -> dict[str, Any]:
    path = _CONFIG_DIR / name
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {path}")
    return data


_agents_cache: dict[str, Any] | None = None
_tasks_cache: dict[str, Any] | None = None


def _agents() -> dict[str, Any]:
    global _agents_cache
    if _agents_cache is None:
        _agents_cache = _load_yaml("agents.yaml")
    return _agents_cache


def _tasks() -> dict[str, Any]:
    global _tasks_cache
    if _tasks_cache is None:
        _tasks_cache = _load_yaml("tasks.yaml")
    return _tasks_cache


def interpolate(template: str, ctx: dict[str, str]) -> str:
    return template.format(**ctx)


def context_from_state(state: PodcasterState) -> dict[str, str]:
    topic = state.get("topic") or ""
    return {
        "topic": topic,
        "current_month": str(state.get("current_month") or ""),
        "current_year": str(state.get("current_year") or ""),
        "male_host": str(state.get("male_host") or "Jone"),
        "female_host": str(state.get("female_host") or "Jane"),
    }


def _normalize_block(s: str) -> str:
    return " ".join(s.split()).strip()


def agent_section(key: str, ctx: dict[str, str]) -> tuple[str, str, str]:
    raw = _agents()[key]
    role = interpolate(_normalize_block(raw["role"]), ctx)
    goal = interpolate(_normalize_block(raw["goal"]), ctx)
    back = raw["backstory"]
    if isinstance(back, dict):
        # YAML folded mapping (rare); join values
        back = " ".join(str(v) for v in back.values() if v)
    backstory = interpolate(_normalize_block(str(back)), ctx)
    return role, goal, backstory


def task_section(key: str, ctx: dict[str, str]) -> tuple[str, str, str]:
    raw = _tasks()[key]
    desc = interpolate(_normalize_block(raw["description"]), ctx)
    expected = interpolate(_normalize_block(raw["expected_output"]), ctx)
    agent_key = str(raw["agent"])
    return desc, expected, agent_key
