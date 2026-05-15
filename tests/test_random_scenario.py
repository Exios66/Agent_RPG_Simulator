from pathlib import Path

import pytest

from agent_rpg.backends.fake import FakeLLMBackend
from agent_rpg.engine import SimulationEngine
from agent_rpg.random_scenario import (
    build_random_scenario,
    list_world_preset_titles,
    refresh_agent_relationships,
)


def test_build_random_scenario_valid():
    s = build_random_scenario(seed=42, num_agents=3, max_rounds=2)
    assert len(s.agents) == 3
    assert s.orchestration.max_rounds == 2 or s.world.max_rounds == 2
    ids = {a.agent_id for a in s.agents}
    for a in s.agents:
        for other in a.relationships:
            if other not in ("narrator", "scene", "crowd"):
                assert other in ids


def test_world_title_selects_preset() -> None:
    title = list_world_preset_titles()[0]
    s = build_random_scenario(seed=0, num_agents=2, max_rounds=1, world_title=title)
    assert s.world.title == title


def test_world_title_unknown_still_preset_from_library() -> None:
    s = build_random_scenario(seed=99, num_agents=2, max_rounds=1, world_title="Not a real preset title")
    assert s.world.title in list_world_preset_titles()


def test_stop_phrase_explicit() -> None:
    s = build_random_scenario(
        seed=1,
        num_agents=2,
        max_rounds=1,
        stop_phrase="ENDX",
        sample_stop_phrase=False,
    )
    assert s.orchestration.stop_phrase == "ENDX"


def test_sample_stop_phrase_eventually_non_null() -> None:
    for seed in range(800):
        s = build_random_scenario(
            seed=seed,
            num_agents=2,
            max_rounds=1,
            stop_phrase="",
            sample_stop_phrase=True,
        )
        if s.orchestration.stop_phrase in ("SCENE_END", "PARLEY_NOW"):
            return
    pytest.fail("expected sampled stop phrase within 800 seeds")


def test_refresh_agent_relationships_reflects_rename() -> None:
    s = build_random_scenario(seed=10, num_agents=2, max_rounds=1)
    a0, a1 = s.agents[0], s.agents[1]
    a0.display_name = "CustomName"
    refresh_agent_relationships(s, seed=123)
    key = a0.agent_id
    assert key in a1.relationships
    assert "CustomName" in a1.relationships[key]


def test_random_scenario_engine_smoke(tmp_path: Path) -> None:
    s = build_random_scenario(seed=7, num_agents=3, max_rounds=2, turn_order="round_robin")
    s.orchestration.enable_thought_phase = False

    def fac(i, msgs):
        return '{"thought":"","say":"ack","directed_at":null}'

    out = SimulationEngine(s).run(FakeLLMBackend(factory=fac), output_dir=tmp_path, run_id="rs7")
    text = (out / "events.jsonl").read_text(encoding="utf-8")
    assert "message" in text
