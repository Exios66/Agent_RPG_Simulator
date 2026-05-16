"""SimulationEngine routing for per-agent OpenRouter backend."""

from __future__ import annotations

import pytest

from agent_rpg.backends.fake import FakeLLMBackend
from agent_rpg.engine import SimulationEngine
from agent_rpg.random_scenario import build_random_scenario


def test_openrouter_agent_without_openrouter_backend_raises() -> None:
    s = build_random_scenario(seed=0, num_agents=2, max_rounds=1)
    for a in s.agents:
        a.backend = "openrouter"
    engine = SimulationEngine(s)
    fac = lambda _i, _m: '{"thought":"","say":"ok","directed_at":null}'
    with pytest.raises(ValueError, match="openrouter_backend"):
        engine.run(FakeLLMBackend(factory=fac), run_id="t1")


def test_openrouter_agent_uses_openrouter_backend() -> None:
    s = build_random_scenario(seed=0, num_agents=2, max_rounds=1, turn_order="round_robin")
    for a in s.agents:
        a.backend = "openrouter"
    engine = SimulationEngine(s)
    calls: list[str] = []

    class Track:
        def generate(self, messages, **kwargs):  # noqa: ANN001
            calls.append(kwargs.get("model_id", ""))
            return '{"thought":"","say":"line","directed_at":null}'

    out = engine.run(
        FakeLLMBackend(factory=lambda _i, _m: "unused"),
        openrouter_backend=Track(),
        run_id="t2",
    )
    assert len(calls) == 2
    assert calls == [a.model_id for a in s.agents]
    assert (out / "events.jsonl").is_file()
