from __future__ import annotations

import json
import random
import uuid
from pathlib import Path

from collections.abc import Callable
from typing import Any

from agent_rpg.backends.base import LLMBackend
from agent_rpg.logging.jsonl import JsonlEventWriter
from agent_rpg.logging.sqlite_store import SqliteEventStore
from agent_rpg.parse import parse_agent_json_response, parse_router_response
from agent_rpg.prompts import build_agent_system_prompt
from agent_rpg.schemas.agent import AgentConfig
from agent_rpg.schemas.events import SimulationEvent
from agent_rpg.schemas.scenario import ScenarioConfig


_JSON_INSTRUCTION = (
    "For this turn, reply with a single JSON object only, with keys "
    '`thought` (string, may be ""), `say` (string, must be non-empty in-character dialogue), '
    "`directed_at` (string agent_id or null)."
)


def _emit(
    writer: JsonlEventWriter,
    sql_store: SqliteEventStore | None,
    event: SimulationEvent,
    on_event: Callable[[SimulationEvent], None] | None = None,
) -> None:
    writer.write(event)
    if sql_store is not None:
        sql_store.insert(event)
    if on_event is not None:
        on_event(event)


def _backend_for_agent(
    agent: AgentConfig,
    default: LLMBackend,
    local_backend: LLMBackend | None,
    openrouter_backend: LLMBackend | None,
) -> LLMBackend:
    if agent.backend == "openrouter":
        if openrouter_backend is None:
            raise ValueError(
                f"Agent {agent.agent_id} uses backend 'openrouter' but no openrouter_backend was passed to SimulationEngine.run()"
            )
        return openrouter_backend
    if agent.backend in ("auto", "hf_inference"):
        return default
    if agent.backend == "transformers_local":
        if local_backend is None:
            raise ValueError(
                f"Agent {agent.agent_id} uses backend 'transformers_local' but no local_backend was passed to SimulationEngine.run()"
            )
        return local_backend
    return default


class SimulationEngine:
    def __init__(self, scenario: ScenarioConfig) -> None:
        self.scenario = scenario

    def run(
        self,
        backend: LLMBackend,
        *,
        local_backend: LLMBackend | None = None,
        openrouter_backend: LLMBackend | None = None,
        run_id: str | None = None,
        output_dir: str | Path | None = None,
        sqlite_path: str | Path | None = None,
        seed: int | None = None,
        on_event: Callable[[SimulationEvent], None] | None = None,
        llm_extras: dict[str, Any] | None = None,
    ) -> Path:
        rid = run_id or str(uuid.uuid4())
        base = Path(output_dir or "runs") / rid
        base.mkdir(parents=True, exist_ok=True)
        jsonl_path = base / "events.jsonl"
        writer = JsonlEventWriter(jsonl_path)
        sql_store: SqliteEventStore | None = None
        if sqlite_path is not None:
            sql_store = SqliteEventStore(Path(sqlite_path))
            sql_store.clear_run(rid)

        try:
            return self._run_loop(
                rid,
                writer,
                sql_store,
                backend,
                local_backend,
                openrouter_backend,
                seed,
                on_event,
                llm_extras,
                base,
            )
        finally:
            writer.close()
            if sql_store is not None:
                sql_store.close()

    def _run_loop(
        self,
        rid: str,
        writer: JsonlEventWriter,
        sql_store: SqliteEventStore | None,
        backend: LLMBackend,
        local_backend: LLMBackend | None,
        openrouter_backend: LLMBackend | None,
        seed: int | None,
        on_event: Callable[[SimulationEvent], None] | None,
        llm_extras: dict[str, Any] | None,
        base: Path,
    ) -> Path:
        rng = random.Random(seed)
        llm_kw = dict(llm_extras or {})

        orch = self.scenario.orchestration
        world = self.scenario.world
        max_rounds = orch.max_rounds or world.max_rounds
        agents = list(self.scenario.agents)

        _emit(
            writer,
            sql_store,
            SimulationEvent(
                run_id=rid,
                round=0,
                agent_id=None,
                event_type="system",
                payload={"message": "simulation_start", "scenario_id": world.scenario_id},
            ),
            on_event,
        )

        transcript: list[str] = []
        stop_phrase = (orch.stop_phrase or "").lower() if orch.stop_phrase else None

        for round_ix in range(max_rounds):
            active_events = [
                e
                for e in world.background_events
                if e.round_trigger is None or e.round_trigger <= round_ix
            ]
            for e in active_events:
                # Log each scripted world event only once — when it first becomes active.
                first_round = 0 if e.round_trigger is None else int(e.round_trigger)
                if round_ix != first_round:
                    continue
                _emit(
                    writer,
                    sql_store,
                    SimulationEvent(
                        run_id=rid,
                        round=round_ix,
                        agent_id=None,
                        event_type="world_event",
                        payload={"event_id": e.id, "description": e.description},
                    ),
                    on_event,
                )

            ev_text = "\n".join(f"- [{e.id}] {e.description}" for e in active_events) or "(none)"
            speakers = self._speaker_order(
                round_ix,
                agents,
                backend,
                local_backend,
                openrouter_backend,
                transcript,
                rng,
                writer,
                sql_store,
                rid,
                on_event,
                llm_kw,
            )

            for agent in speakers:
                thought_on = orch.enable_thought_phase
                if agent.include_thought_phase is not None:
                    thought_on = agent.include_thought_phase

                system = build_agent_system_prompt(
                    self.scenario,
                    agent,
                    active_events_text=ev_text,
                    round_index=round_ix,
                )
                # ``transcript[-0:]`` is the full list in Python (``-0`` == ``0``), so ``memory_turns == 0``
                # must not use a negative slice — it would mean "unbounded memory" instead of none.
                keep_lines = orch.memory_turns * 4
                if not transcript:
                    convo = "(silence so far)"
                elif keep_lines <= 0:
                    convo = ""
                else:
                    convo = "\n".join(transcript[-keep_lines:])
                user_tail = (
                    f"Conversation so far:\n{convo}\n\n{_JSON_INSTRUCTION}\n"
                    f"If you make a major decision, prefix an important line inside `say` with DECISION: "
                )
                messages: list[dict[str, str]] = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_tail},
                ]
                model_id = agent.model_id
                gen_backend = _backend_for_agent(agent, backend, local_backend, openrouter_backend)
                raw = gen_backend.generate(
                    messages,
                    model_id=model_id,
                    max_new_tokens=agent.max_new_tokens,
                    temperature=agent.temperature,
                    top_p=agent.top_p,
                    **llm_kw,
                )
                parsed = parse_agent_json_response(raw)
                if parsed.parse_error:
                    _emit(
                        writer,
                        sql_store,
                        SimulationEvent(
                            run_id=rid,
                            round=round_ix,
                            agent_id=agent.agent_id,
                            event_type="error",
                            payload={"stage": "parse", "error": parsed.parse_error, "raw": parsed.raw},
                        ),
                        on_event,
                    )

                if thought_on and agent.log_thoughts and parsed.thought:
                    _emit(
                        writer,
                        sql_store,
                        SimulationEvent(
                            run_id=rid,
                            round=round_ix,
                            agent_id=agent.agent_id,
                            event_type="thought",
                            payload={"text": parsed.thought},
                        ),
                        on_event,
                    )

                _emit(
                    writer,
                    sql_store,
                    SimulationEvent(
                        run_id=rid,
                        round=round_ix,
                        agent_id=agent.agent_id,
                        event_type="message",
                        payload={
                            "text": parsed.say,
                            "directed_at": parsed.directed_at,
                            "parse_error": parsed.parse_error,
                        },
                    ),
                    on_event,
                )

                line = f"{agent.display_name} ({agent.agent_id}): {parsed.say}"
                transcript.append(line)

                if stop_phrase and stop_phrase in parsed.say.lower():
                    _emit(
                        writer,
                        sql_store,
                        SimulationEvent(
                            run_id=rid,
                            round=round_ix,
                            agent_id=None,
                            event_type="system",
                            payload={"message": "stopped_by_phrase", "phrase": orch.stop_phrase},
                        ),
                        on_event,
                    )
                    return base

        _emit(
            writer,
            sql_store,
            SimulationEvent(
                run_id=rid,
                round=max_rounds - 1 if max_rounds else 0,
                agent_id=None,
                event_type="system",
                payload={"message": "simulation_end", "reason": "max_rounds"},
            ),
            on_event,
        )
        return base

    def _speaker_order(
        self,
        round_ix: int,
        agents: list[AgentConfig],
        backend: LLMBackend,
        local_backend: LLMBackend | None,
        openrouter_backend: LLMBackend | None,
        transcript: list[str],
        rng: random.Random,
        writer: JsonlEventWriter,
        sql_store: SqliteEventStore | None,
        run_id: str,
        on_event: Callable[[SimulationEvent], None] | None,
        llm_kw: dict[str, Any],
    ) -> list[AgentConfig]:
        orch = self.scenario.orchestration
        order = orch.turn_order
        if order == "round_robin":
            return agents
        if order == "random":
            shuffled = list(agents)
            rng.shuffle(shuffled)
            return shuffled
        # reactive
        allowed = {a.agent_id for a in agents}
        router_model = orch.reactive_router_model_id or agents[0].model_id
        router_agent = agents[0]
        router_gen = _backend_for_agent(router_agent, backend, local_backend, openrouter_backend)
        sys = (
            "You are a scene director. Given the list of agent ids and the latest transcript, "
            "choose exactly one agent who should speak next this turn. "
            'Reply with JSON only: {"next_agent_id":"<id>"}.'
        )
        user = {
            "agent_ids": list(allowed),
            "transcript_tail": "\n".join(transcript[-12:]),
        }

        raw = router_gen.generate(
            [{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user)}],
            model_id=router_model,
            max_new_tokens=64,
            temperature=0.2,
            **llm_kw,
        )
        nxt = parse_router_response(raw, allowed)
        _emit(
            writer,
            sql_store,
            SimulationEvent(
                run_id=run_id,
                round=round_ix,
                agent_id=None,
                event_type="router",
                payload={"raw": raw, "chosen": nxt},
            ),
            on_event,
        )
        if nxt is None:
            return agents
        chosen = [a for a in agents if a.agent_id == nxt]
        rest = [a for a in agents if a.agent_id != nxt]
        return chosen + rest
