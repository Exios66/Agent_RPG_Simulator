import json

import pytest
from pydantic import ValidationError

from agent_rpg.loader import load_scenario, scenario_json_schema
from agent_rpg.schemas.scenario import ScenarioConfig


def test_load_minimal_examples():
    s = load_scenario("examples/scenarios/minimal.yaml")
    assert s.world.scenario_id == "minimal"
    assert len(s.agents) == 2


def test_scenario_json_schema_has_title():
    schema = scenario_json_schema()
    assert "ScenarioConfig" in json.dumps(schema)


def test_invalid_relationship_rejected():
    raw = {
        "world": {
            "scenario_id": "x",
            "title": "t",
            "max_rounds": 1,
        },
        "agents": [
            {
                "agent_id": "a1",
                "display_name": "A",
                "relationships": {"ghost": "nope"},
            },
        ],
    }
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(raw)
