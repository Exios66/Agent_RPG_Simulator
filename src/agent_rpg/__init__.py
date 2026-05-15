"""Multi-agent LLM RPG scenario simulator.

Public names are loaded on first access so one optional submodule failing does not
block the rest (and ``from agent_rpg import SimulationEngine`` gets a clear error).
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "SimulationEngine",
    "load_scenario",
    "scenario_json_schema",
    "ReportBuilder",
    "build_random_scenario",
    "find_repo_root",
    "DEFAULT_INSTRUCT_MODEL_ID",
    "SMALL_INSTRUCT_MODELS",
    "labels_by_id",
    "model_ids_for_widgets",
    "assign_models_to_agents",
    "set_router_model_if_reactive",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "SimulationEngine": ("agent_rpg.engine", "SimulationEngine"),
    "load_scenario": ("agent_rpg.loader", "load_scenario"),
    "scenario_json_schema": ("agent_rpg.loader", "scenario_json_schema"),
    "ReportBuilder": ("agent_rpg.reporting.builder", "ReportBuilder"),
    "build_random_scenario": ("agent_rpg.random_scenario", "build_random_scenario"),
    "find_repo_root": ("agent_rpg.repo_root", "find_repo_root"),
    "DEFAULT_INSTRUCT_MODEL_ID": ("agent_rpg.model_catalog", "DEFAULT_INSTRUCT_MODEL_ID"),
    "SMALL_INSTRUCT_MODELS": ("agent_rpg.model_catalog", "SMALL_INSTRUCT_MODELS"),
    "labels_by_id": ("agent_rpg.model_catalog", "labels_by_id"),
    "model_ids_for_widgets": ("agent_rpg.model_catalog", "model_ids_for_widgets"),
    "assign_models_to_agents": ("agent_rpg.multi_model", "assign_models_to_agents"),
    "set_router_model_if_reactive": ("agent_rpg.multi_model", "set_router_model_if_reactive"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        mod_name, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, attr)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
