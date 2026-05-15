from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import TypeAdapter

from agent_rpg.schemas.scenario import ScenarioConfig


def scenario_json_schema() -> dict:
    """JSON Schema for `ScenarioConfig` (tooling / editors)."""
    return ScenarioConfig.model_json_schema()


def load_scenario(path: str | Path) -> ScenarioConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return TypeAdapter(ScenarioConfig).validate_python(raw)
