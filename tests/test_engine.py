from pathlib import Path

import pytest

from agent_rpg.backends.fake import FakeLLMBackend
from agent_rpg.engine import SimulationEngine
from agent_rpg.loader import load_scenario
from agent_rpg.logging.jsonl import iter_events_jsonl
from agent_rpg.schemas.world import BackgroundEvent


def _factory(i: int, messages: list[dict[str, str]]) -> str:
    # First calls may be router in reactive scenarios; minimal uses round_robin only.
    if "scene director" in messages[0].get("content", "").lower():
        return '{"next_agent_id":"alice"}'
    return (
        '{"thought":"thinking","say":"Hello round","directed_at":null}'
        if i % 2 == 0
        else '{"thought":"","say":"Reply here SCENE_END","directed_at":"alice"}'
    )


def test_engine_round_robin_stop_phrase(tmp_path: Path):
    s = load_scenario("examples/scenarios/minimal.yaml")
    s.orchestration.stop_phrase = "SCENE_END"
    s.orchestration.max_rounds = 10
    s.world.max_rounds = 10
    backend = FakeLLMBackend(factory=_factory)
    eng = SimulationEngine(s)
    out = eng.run(backend, output_dir=tmp_path, run_id="r1")
    events = iter_events_jsonl(out / "events.jsonl")
    types = [e.event_type for e in events]
    assert "message" in types
    assert any(e.event_type == "system" and e.payload.get("message") == "stopped_by_phrase" for e in events)


def test_engine_sqlite_mirror(tmp_path: Path):
    s = load_scenario("examples/scenarios/minimal.yaml")
    s.orchestration.enable_thought_phase = False
    s.orchestration.max_rounds = 1
    s.world.max_rounds = 1

    def fac(i: int, msgs: list[dict[str, str]]) -> str:
        return '{"thought":"","say":"ok","directed_at":null}'

    backend = FakeLLMBackend(factory=fac)
    db = tmp_path / "e.sqlite"
    eng = SimulationEngine(s)
    eng.run(backend, output_dir=tmp_path, run_id="sql", sqlite_path=db)
    assert db.exists()
    import sqlite3

    con = sqlite3.connect(db)
    n = con.execute("select count(*) from events").fetchone()[0]
    assert n >= 1


def test_local_backend_missing_raises(tmp_path: Path):
    s = load_scenario("examples/scenarios/minimal.yaml")
    s.agents[0].backend = "transformers_local"
    backend = FakeLLMBackend(
        responses=['{"thought":"","say":"x","directed_at":null}'] * 20,
    )
    eng = SimulationEngine(s)
    with pytest.raises(ValueError, match="local_backend"):
        eng.run(backend, output_dir=tmp_path, run_id="x")


def test_world_event_emitted_once_when_event_first_activates(tmp_path: Path):
    """Background beats should not re-log every round after their trigger."""
    s = load_scenario("examples/scenarios/minimal.yaml")
    s.world.background_events = [
        BackgroundEvent(id="e_open", description="Doors slam open.", round_trigger=0),
        BackgroundEvent(id="e_mid", description="A bell rings.", round_trigger=1),
    ]
    s.orchestration.max_rounds = 4
    s.world.max_rounds = 4
    s.orchestration.enable_thought_phase = False

    def fac(_i: int, _msgs: list[dict[str, str]]) -> str:
        return '{"thought":"","say":"ok","directed_at":null}'

    backend = FakeLLMBackend(factory=fac)
    SimulationEngine(s).run(backend, output_dir=tmp_path, run_id="we1")
    events = iter_events_jsonl(tmp_path / "we1" / "events.jsonl")
    world_ev = [e for e in events if e.event_type == "world_event"]
    assert len(world_ev) == 2
    by_id = {e.payload["event_id"]: e.round for e in world_ev}
    assert by_id["e_open"] == 0 and by_id["e_mid"] == 1


def test_reactive_turn_order_calls_router(tmp_path: Path):
    s = load_scenario("examples/scenarios/minimal.yaml")
    s.orchestration.turn_order = "reactive"
    s.orchestration.max_rounds = 1
    s.world.max_rounds = 1
    s.orchestration.enable_thought_phase = False
    calls: list[str] = []

    def fac(i: int, msgs: list[dict[str, str]]) -> str:
        sysm = msgs[0].get("content", "")
        if "scene director" in sysm.lower():
            calls.append("router")
            return '{"next_agent_id":"bob"}'
        calls.append("agent")
        return '{"thought":"","say":"hi","directed_at":null}'

    backend = FakeLLMBackend(factory=fac)
    SimulationEngine(s).run(backend, output_dir=tmp_path, run_id="rx")
    assert "router" in calls
    assert "agent" in calls


def test_memory_turns_zero_omits_prior_transcript(tmp_path: Path):
    """``memory_turns == 0`` must not expand to the full history (Python ``seq[-0:]`` is ``seq[:]``)."""
    s = load_scenario("examples/scenarios/minimal.yaml")
    s.orchestration.memory_turns = 0
    s.orchestration.enable_thought_phase = False
    s.orchestration.max_rounds = 2
    s.world.max_rounds = 2

    def fac(i: int, _msgs: list[dict[str, str]]) -> str:
        if i < 2:
            return '{"thought":"","say":"EARLY_UNIQUE_LINE","directed_at":null}'
        return '{"thought":"","say":"ok","directed_at":null}'

    backend = FakeLLMBackend(factory=fac)
    SimulationEngine(s).run(backend, output_dir=tmp_path, run_id="m0")
    assert len(backend.calls) >= 3
    user3 = backend.calls[2][0][1].get("content", "")
    assert "EARLY_UNIQUE_LINE" not in user3


def test_engine_parse_error_emits_error_event(tmp_path: Path):
    s = load_scenario("examples/scenarios/minimal.yaml")
    s.orchestration.enable_thought_phase = False
    s.orchestration.max_rounds = 1
    s.world.max_rounds = 1

    backend = FakeLLMBackend(responses=["not valid json"] * 2)
    eng = SimulationEngine(s)
    eng.run(backend, output_dir=tmp_path, run_id="pe1")
    events = iter_events_jsonl(tmp_path / "pe1" / "events.jsonl")
    errors = [e for e in events if e.event_type == "error"]
    assert len(errors) == 2
    assert all(e.payload["stage"] == "parse" for e in errors)
    assert all(e.payload["error"] == "invalid_json" for e in errors)
    messages = [e for e in events if e.event_type == "message"]
    assert len(messages) == 2
    assert all(m.payload["parse_error"] == "invalid_json" for m in messages)


def test_engine_on_event_callback_receives_events(tmp_path: Path):
    s = load_scenario("examples/scenarios/minimal.yaml")
    s.orchestration.enable_thought_phase = False
    s.orchestration.max_rounds = 1
    s.world.max_rounds = 1

    backend = FakeLLMBackend(
        responses=['{"thought":"","say":"ok","directed_at":null}'] * 2,
    )
    received: list[str] = []

    def on_event(ev):
        received.append(ev.event_type)

    SimulationEngine(s).run(
        backend,
        output_dir=tmp_path,
        run_id="cb1",
        on_event=on_event,
    )
    assert "system" in received
    assert "message" in received


def test_engine_random_turn_order_is_seeded(tmp_path: Path):
    s = load_scenario("examples/scenarios/minimal.yaml")
    s.orchestration.turn_order = "random"
    s.orchestration.enable_thought_phase = False
    s.orchestration.max_rounds = 1
    s.world.max_rounds = 1

    def fac(_i: int, _msgs: list[dict[str, str]]) -> str:
        return '{"thought":"","say":"ok","directed_at":null}'

    def first_agent_id(seed: int) -> str:
        backend = FakeLLMBackend(factory=fac)
        SimulationEngine(s).run(
            backend,
            output_dir=tmp_path,
            run_id=f"rnd{seed}",
            seed=seed,
        )
        first_line = backend.calls[0][0][0]["content"].splitlines()[0]
        return "bob" if first_line.startswith("You are Bob") else "alice"

    assert first_agent_id(42) == first_agent_id(42)
    orders = {first_agent_id(seed) for seed in range(30)}
    assert len(orders) == 2


def test_memory_turns_truncates_prior_transcript(tmp_path: Path):
    s = load_scenario("examples/scenarios/minimal.yaml")
    s.orchestration.memory_turns = 1
    s.orchestration.enable_thought_phase = False
    s.orchestration.max_rounds = 4
    s.world.max_rounds = 4

    def fac(i: int, _msgs: list[dict[str, str]]) -> str:
        return f'{{"thought":"","say":"LINE_{i}","directed_at":null}}'

    backend = FakeLLMBackend(factory=fac)
    SimulationEngine(s).run(backend, output_dir=tmp_path, run_id="m1")
    # Round 3, first speaker (call index 6): memory_turns=1 keeps only the last 4 lines.
    user_content = backend.calls[6][0][1].get("content", "")
    assert "LINE_0" not in user_content
    assert "LINE_1" not in user_content
    assert "LINE_6" in user_content or "LINE_5" in user_content
