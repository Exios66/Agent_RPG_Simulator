"""Procedurally generate valid :class:`ScenarioConfig` instances for experiments and demos."""

from __future__ import annotations

import random
import uuid
from typing import Literal

from agent_rpg.schemas.agent import AgentConfig
from agent_rpg.schemas.orchestration import OrchestrationConfig
from agent_rpg.schemas.scenario import ScenarioConfig
from agent_rpg.schemas.world import BackgroundEvent, WorldConfig

_FIRST_NAMES = (
    "Mara", "Jax", "Silas", "Pari", "Ilo", "Vesa", "Oru", "Mei", "Kira", "Dorn",
    "Elara", "Finn", "Gwen", "Hugo", "Inez", "Juno",
)
_LAST_NAMES = (
    "Keen", "Ironhand", "Rook", "Ash", "Vale", "Storm", "Grey", "Bloom", "Frost", "Reed",
)
_ARCHETYPES = (
    "skeptical scholar", "hot-headed sellsword", "charming rogue", "pragmatic engineer",
    "impatient merchant", "cautious elder", "idealistic youth", "cynical veteran",
    "secretive spy", "devout healer", "greedy alchemist", "proud noble",
)
_OCCUPATIONS = (
    "innkeeper", "guard", "smuggler", "notary", "cartographer", "blacksmith", "bard",
    "dockworker", "archivist", "mercenary", "herbalist", "sailor", "tutor", "beadle",
)
_REL_TEMPLATES = (
    "distrusts {o}", "owes {o} a favor", "secretly admires {o}", "rival since childhood with {o}",
    "business partner of {o}", "tries to impress {o}", "fears {o}'s temper", "protective of {o}",
    "envies {o}'s reputation", "owes money to {o}", "spies on {o}", "confides in {o}",
)
_GOALS = (
    "Secure a fair outcome without bloodshed.",
    "Learn a critical secret before anyone else.",
    "Protect your reputation at all costs.",
    "Leave with coin or leverage.",
    "De-escalate but do not appear weak.",
    "Find out who is lying before the hour ends.",
)
_EVENT_TEMPLATES = (
    "A bell tolls; the crowd murmurs.",
    "Torches flicker as wind gusts through the hall.",
    "A messenger arrives breathless with news from the gate.",
    "Someone drops a coin purse; eyes dart.",
    "Distant thunder; rain begins against the shutters.",
    "A dog barks; a child screams playfully outside.",
    "The smell of smoke — cookfire or warning?",
    "A seal on a letter appears freshly broken.",
)
_SETTINGS = (
    {
        "title": "Storm over the trade docks",
        "setting": "Salt-slick piers, stacked crates, lanterns swaying in wind.",
        "backstory": "A disputed shipment and missing manifest have drawn rivals into the same warehouse.",
        "worldbuilding_notes": "Firearms are rare; blades and wits rule the night.",
    },
    {
        "title": "Council under the old oak",
        "setting": "Village commons at dusk; farmers and traders circle a splintered bench.",
        "backstory": "The well ran dry; blame ricochets between upstream farms and the mill.",
        "worldbuilding_notes": "Tradition forbids drawing steel under the oak.",
    },
    {
        "title": "Archive of sealed letters",
        "setting": "Marble corridors, dust motes, echoing footsteps.",
        "backstory": "A forbidden index volume vanished; each suspect had access last week.",
        "worldbuilding_notes": "Magic lights candles but cannot unburn parchment.",
    },
    {
        "title": "Festival night market",
        "setting": "Paper lanterns, skewer smoke, a fortune-teller's tent flapping.",
        "backstory": "A relic was pickpocketed during the parade; accusations fly between troupes.",
        "worldbuilding_notes": "City watch patrols in pairs every quarter hour.",
    },
    {
        "title": "Mountain pass negotiation",
        "setting": "Thin air, rope bridges, two banners snapping in cold wind.",
        "backstory": "Tolls doubled overnight; caravans refuse to move until terms change.",
        "worldbuilding_notes": "Avalanche risk if voices rise too loud — local superstition.",
    },
)


def build_random_scenario(
    *,
    seed: int | None = None,
    num_agents: int | None = None,
    max_rounds: int | None = None,
    model_id: str = "HuggingFaceH4/zephyr-7b-beta",
    turn_order: Literal["round_robin", "random", "reactive"] | None = None,
) -> ScenarioConfig:
    """
    Build a new randomized but schema-valid scenario.

    Relationships only reference other agents in the same scenario (plus optional narrator).
    """
    rng = random.Random(seed)
    n = num_agents if num_agents is not None else rng.randint(2, 5)
    n = max(2, min(n, 8))

    mr = max_rounds if max_rounds is not None else rng.randint(2, 4)
    mr = max(1, min(mr, 8))

    pick = rng.choice(_SETTINGS)
    scenario_id = f"rand_{uuid.uuid4().hex[:10]}"

    agent_ids: list[str] = [f"agent_{i}" for i in range(n)]
    id_to_name: dict[str, str] = {}

    for i, aid in enumerate(agent_ids):
        fn = rng.choice(_FIRST_NAMES)
        ln = rng.choice(_LAST_NAMES)
        display = f"{fn} {ln}" if rng.random() > 0.15 else fn
        id_to_name[aid] = display

    agents: list[AgentConfig] = []
    for aid in agent_ids:
        others = [x for x in agent_ids if x != aid]
        rels: dict[str, str] = {}
        if others:
            k = rng.randint(1, min(3, len(others)))
            for other in rng.sample(others, k=k):
                tpl = rng.choice(_REL_TEMPLATES)
                oname = id_to_name[other]
                rels[other] = tpl.format(o=oname)
        if rng.random() < 0.25:
            rels["narrator"] = rng.choice(
                (
                    "the unseen chorus pressures you to confess",
                    "fate feels unusually attentive",
                )
            )
        agents.append(
            AgentConfig(
                agent_id=aid,
                display_name=id_to_name[aid],
                archetype=rng.choice(_ARCHETYPES),
                occupation=rng.choice(_OCCUPATIONS),
                prompt_template_id="default",
                prompt_variables={"goal": rng.choice(_GOALS)},
                model_id=model_id,
                backend="auto",
                temperature=round(rng.uniform(0.35, 0.95), 2),
                max_new_tokens=rng.choice([128, 192, 256, 320]),
                top_p=round(rng.uniform(0.85, 0.98), 3) if rng.random() > 0.3 else None,
                relationships=rels,
                log_thoughts=rng.random() > 0.35,
                include_thought_phase=None if rng.random() > 0.4 else rng.choice([True, False]),
            )
        )

    n_events = rng.randint(0, min(4, mr + 1))
    events: list[BackgroundEvent] = []
    for e in range(n_events):
        rid = f"evt_{e}_{rng.randint(100, 999)}"
        desc = rng.choice(_EVENT_TEMPLATES)
        trig: int | None
        if rng.random() < 0.35:
            trig = None
        else:
            trig = rng.randint(0, max(0, mr - 1))
        events.append(BackgroundEvent(id=rid, description=desc, round_trigger=trig))

    order = turn_order or rng.choice(["round_robin", "random", "reactive"])
    thought = rng.random() > 0.45
    stop = rng.choice([None, None, None, "SCENE_END", "PARLEY_NOW"]) if rng.random() > 0.7 else None

    world = WorldConfig(
        scenario_id=scenario_id,
        title=pick["title"],
        version="1",
        setting=pick["setting"],
        backstory=pick["backstory"],
        worldbuilding_notes=pick["worldbuilding_notes"],
        background_events=events,
        max_rounds=mr,
        round_duration_narrative=rng.choice(
            ("Minutes feel like hours.", "Each exchange burns social capital.", "Time is slipping.")
        ),
        default_occupations=rng.sample(_OCCUPATIONS, k=min(3, len(_OCCUPATIONS))),
    )

    orch = OrchestrationConfig(
        turn_order=order,
        max_rounds=mr,
        enable_thought_phase=thought,
        memory_turns=rng.choice([12, 16, 20, 24]),
        stop_phrase=stop,
        reactive_router_model_id=model_id if order == "reactive" else None,
    )

    return ScenarioConfig(world=world, agents=agents, orchestration=orch)
