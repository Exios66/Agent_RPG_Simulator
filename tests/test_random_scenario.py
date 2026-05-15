from pathlib import Path

from agent_rpg.backends.fake import FakeLLMBackend
from agent_rpg.engine import SimulationEngine
from agent_rpg.random_scenario import build_random_scenario


def test_build_random_scenario_valid():
    s = build_random_scenario(seed=42, num_agents=3, max_rounds=2)
    assert len(s.agents) == 3
    assert s.orchestration.max_rounds == 2 or s.world.max_rounds == 2
    ids = {a.agent_id for a in s.agents}
    for a in s.agents:
        for other in a.relationships:
            if other not in ("narrator", "scene", "crowd"):
                assert other in ids


def test_random_scenario_engine_smoke(tmp_path: Path) -> None:
    s = build_random_scenario(seed=7, num_agents=3, max_rounds=2, turn_order="round_robin")
    s.orchestration.enable_thought_phase = False

    def fac(i, msgs):
        return '{"thought":"","say":"ack","directed_at":null}'

    out = SimulationEngine(s).run(FakeLLMBackend(factory=fac), output_dir=tmp_path, run_id="rs7")
    text = (out / "events.jsonl").read_text(encoding="utf-8")
    assert "message" in text
