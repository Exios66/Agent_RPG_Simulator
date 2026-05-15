"""Multi-agent LLM RPG scenario simulator."""

from agent_rpg.engine import SimulationEngine
from agent_rpg.loader import load_scenario, scenario_json_schema
from agent_rpg.model_catalog import DEFAULT_INSTRUCT_MODEL_ID, SMALL_INSTRUCT_MODELS, labels_by_id, model_ids_for_widgets
from agent_rpg.multi_model import assign_models_to_agents, set_router_model_if_reactive
from agent_rpg.random_scenario import build_random_scenario
from agent_rpg.reporting.builder import ReportBuilder
from agent_rpg.repo_root import find_repo_root

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
