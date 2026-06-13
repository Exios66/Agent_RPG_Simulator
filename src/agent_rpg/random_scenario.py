"""Procedurally generate valid :class:`ScenarioConfig` instances for experiments and demos."""

from __future__ import annotations

import random
import uuid
from typing import Literal

from agent_rpg.model_catalog import DEFAULT_INSTRUCT_MODEL_ID
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

# Rich, scene-specific beats (shown once when they trigger). Tied to `events_key` on each setting.
_EVENT_LIBRARY: dict[str, tuple[str, ...]] = {
    "trade_docks": (
        "A customs chalk-mark appears on three crates that were unmarked at dawn — someone is "
        "quietly rewriting who owns what, and the stevedores have stopped work to watch.",
        "A tugboat's steam whistle slices through the argument; a foreman shouts that the manifest "
        "copy in the harbormaster's office no longer matches the one pinned inside this warehouse.",
        "Rain begins hammering the tin roof; a leak opens above the disputed pallet, and everyone "
        "must decide in the next few minutes whether to move the cargo — or leave it as evidence.",
        "Two rival company seals are pressed into the same wax on a bill of lading; whoever "
        "acknowledges it first may inherit liability — or leverage.",
        "A dockworker whispers that a night watchman was paid to look away when a specific crate "
        "was swapped; the watchman is here now, hands shaking, still wearing the wrong boots.",
        "Lanterns gutter as wind slams a loose door; in the sudden dark, someone counts crates aloud "
        "— and the count does not match what the merchant swore ten minutes ago.",
    ),
    "village_council": (
        "A farmer slams a dried gourd on the bench: upstream ditches were diverted last night, "
        "and the millpond is refilling while the commons well stays empty — accusation hangs "
        "in the air like smoke.",
        "The miller's apprentice admits under pressure that a sealed order arrived from the "
        "county seat — nobody was supposed to open it until after this meeting.",
        "Children drag a dead fish from the stream; it smells of lamp oil, not rot — evidence "
        "that someone flushed something upstream they did not want found.",
        "An elder reminds the circle that oaths spoken under the oak bind families for a "
        "generation; several listeners trade looks, calculating who gains if tradition is invoked.",
        "A trader offers to fund a new well if blame is named tonight; the offer feels less like "
        "charity and more like purchasing a verdict.",
    ),
    "archive": (
        "A candle gutter reveals a bootprint in dust where no scribe should have walked after "
        "curfew — the print points toward the restricted stacks, not the exit.",
        "A reader calls out that two call numbers in the index were altered with different ink; "
        "the older ink matches a pen still on the chief archivist's desk.",
        "A sealed ribbon on a folio has been cut and re-melted clumsily; whoever did it was in a "
        "hurry and may still be carrying wax crumbs in their cuffs.",
        "Footsteps echo on the marble mezzanine — someone is pacing where only senior clerks "
        "may go, and the echo suggests they are listening to your debate below.",
        "A scrap of marginalia falls from a returned volume: it names a blackmail price and a "
        "deadline at moonrise — tonight.",
    ),
    "night_market": (
        "A fortune-teller's tent flap snaps open; inside, a stolen relic's cloth wrapping lies "
        "empty — the thief may still be in the crowd, judging who flees first.",
        "Drums from the parade shift tempo; the watch's whistle answers, and hawkers begin "
        "lowering shutters as if they were warned this hour would turn ugly.",
        "A pickpocket is caught with two purses — one belongs to a troupe leader, the other to "
        "a noble's courier; returning the wrong purse first could spark a duel.",
        "Skewer smoke thickens; someone screams that a vendor's scale weights are false — "
        "everyone with debt at that stall suddenly has a stake in the outcome.",
        "Paper lanterns sway as a sudden breeze carries playing cards into the air; one card "
        "bears a wax seal that matches the missing relic's crate.",
    ),
    "mountain_pass": (
        "Ice cracks somewhere above the pass; an old guide insists lowered voices are not "
        "superstition — avalanches answer to pride as much as to sound.",
        "A second banner party arrives with a duplicate toll charter, stamped the same morning; "
        "only one caravan can be lawful, and both captains demand the bridge first.",
        "A rope bridge handrail frays under inspection; crossing tonight means trusting whoever "
        "volunteers to go last — or who insists on going first.",
        "A courier collapses with frostbitten fingers; their satchel holds a folded map with a "
        "secret detour sketched in charcoal — and a smear of blood as if someone tried to erase it.",
        "Caravan guards begin counting arrows; the count becomes a public challenge about who "
        "plans to enforce which version of the toll by morning.",
    ),
    "generic": (
        "A bell tolls; the crowd's murmur sharpens into side-bargains — alliances are being "
        "re-priced in whispers before the next speaker takes the floor.",
        "Torches gutter in a gust; in the dim half-second, two people reach for the same document "
        "— neither will admit what they hoped to hide.",
        "A messenger arrives breathless, bearing a folded note marked 'read aloud'; whoever opens "
        "it may control the narrative for everyone present.",
        "Someone's coin purse hits the stones; coins spin, and for a heartbeat every eye maps "
        "who moves first — generosity, greed, or guilt.",
        "Distant thunder rolls; shutters bang, and someone suggests postponing — another calls "
        "that delay a tactic, not prudence.",
        "The smell of smoke tightens throats; cookfire or warning becomes a question of whom you "
        "trust to answer honestly.",
        "A seal on a letter looks freshly broken; the wax is still warm under a thumb, and "
        "fingers point before minds catch up.",
    ),
}

_SETTINGS = (
    {
        "title": "Storm over the trade docks",
        "events_key": "trade_docks",
        "setting": "Salt-slick piers, stacked crates, lanterns swaying in wind.",
        "backstory": "A disputed shipment and missing manifest have drawn rivals into the same warehouse.",
        "worldbuilding_notes": "Firearms are rare; blades and wits rule the night.",
    },
    {
        "title": "Council under the old oak",
        "events_key": "village_council",
        "setting": "Village commons at dusk; farmers and traders circle a splintered bench.",
        "backstory": "The well ran dry; blame ricochets between upstream farms and the mill.",
        "worldbuilding_notes": "Tradition forbids drawing steel under the oak.",
    },
    {
        "title": "Archive of sealed letters",
        "events_key": "archive",
        "setting": "Marble corridors, dust motes, echoing footsteps.",
        "backstory": "A forbidden index volume vanished; each suspect had access last week.",
        "worldbuilding_notes": "Magic lights candles but cannot unburn parchment.",
    },
    {
        "title": "Festival night market",
        "events_key": "night_market",
        "setting": "Paper lanterns, skewer smoke, a fortune-teller's tent flapping.",
        "backstory": "A relic was pickpocketed during the parade; accusations fly between troupes.",
        "worldbuilding_notes": "City watch patrols in pairs every quarter hour.",
    },
    {
        "title": "Mountain pass negotiation",
        "events_key": "mountain_pass",
        "setting": "Thin air, rope bridges, two banners snapping in cold wind.",
        "backstory": "Tolls doubled overnight; caravans refuse to move until terms change.",
        "worldbuilding_notes": "Avalanche risk if voices rise too loud — local superstition.",
    },
)


def _sample_background_events(rng: random.Random, events_key: str, max_rounds: int) -> list[BackgroundEvent]:
    pool = list(_EVENT_LIBRARY.get(events_key, _EVENT_LIBRARY["generic"]))
    mr = max(1, max_rounds)
    # Enough beats to punctuate most runs without repeating the same line twice in one scenario.
    n = rng.randint(2, min(len(pool), mr + 1)) if mr >= 2 else rng.randint(0, min(2, len(pool)))
    if n <= 0:
        return []
    chosen = rng.sample(pool, k=n)
    events: list[BackgroundEvent] = []
    for j, desc in enumerate(chosen):
        rid = f"evt_{j}_{rng.randint(1000, 9999)}"
        if j == 0:
            trig: int | None = 0 if rng.random() < 0.55 else rng.randint(0, max(0, mr - 1))
        else:
            trig = rng.randint(0, max(0, mr - 1))
        events.append(BackgroundEvent(id=rid, description=desc, round_trigger=trig))
    return events


def list_world_preset_titles() -> list[str]:
    """Titles of built-in procedural world premises (for UI dropdowns)."""
    return [s["title"] for s in _SETTINGS]


def refresh_agent_relationships(scenario: ScenarioConfig, *, seed: int | None = None) -> None:
    """Rebuild `relationships` prose from current `display_name`s (e.g. after manual renames)."""
    rng = random.Random(seed)
    agent_ids = [a.agent_id for a in scenario.agents]
    id_to_name = {a.agent_id: a.display_name for a in scenario.agents}
    for ag in scenario.agents:
        others = [x for x in agent_ids if x != ag.agent_id]
        rels: dict[str, str] = {}
        if others:
            k = rng.randint(1, min(3, len(others)))
            for other in rng.sample(others, k=k):
                tpl = rng.choice(_REL_TEMPLATES)
                rels[other] = tpl.format(o=id_to_name[other])
        if rng.random() < 0.25:
            rels["narrator"] = rng.choice(
                (
                    "the unseen chorus pressures you to confess",
                    "fate feels unusually attentive",
                )
            )
        ag.relationships = rels


# Exposed for notebook dropdowns (same tuples as procedural generation).
ARCHETYPE_OPTIONS: tuple[str, ...] = _ARCHETYPES
OCCUPATION_OPTIONS: tuple[str, ...] = _OCCUPATIONS
GOAL_OPTIONS: tuple[str, ...] = _GOALS


def build_random_scenario(
    *,
    seed: int | None = None,
    num_agents: int | None = None,
    max_rounds: int | None = None,
    model_id: str | None = None,
    turn_order: Literal["round_robin", "random", "reactive"] | None = None,
    world_title: str | None = None,
    enable_thought_phase: bool | None = None,
    memory_turns: int | None = None,
    stop_phrase: str = "",
    sample_stop_phrase: bool = False,
) -> ScenarioConfig:
    """
    Build a new randomized but schema-valid scenario.

    Relationships only reference other agents in the same scenario (plus optional narrator).

    ``world_title``: if set to one of :func:`list_world_preset_titles`, that premise is used;
    otherwise a random premise is chosen.

    ``enable_thought_phase`` / ``memory_turns``: when ``None``, values are sampled. Pass explicit
    values from a UI to pin them.

    ``stop_phrase``: explicit phrase (e.g. ``SCENE_END``) or empty string for none.
    ``sample_stop_phrase``: if True, occasionally samples a legacy random stop phrase (ignored
    when ``stop_phrase`` is non-empty).
    """
    rng = random.Random(seed)
    mid = model_id or DEFAULT_INSTRUCT_MODEL_ID
    n = num_agents if num_agents is not None else rng.randint(2, 5)
    n = max(2, min(n, 8))

    mr = max_rounds if max_rounds is not None else rng.randint(2, 4)
    mr = max(1, min(mr, 8))

    pick = rng.choice(_SETTINGS)
    if world_title:
        for s in _SETTINGS:
            if s["title"] == world_title.strip():
                pick = s
                break
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
                model_id=mid,
                backend="auto",
                temperature=round(rng.uniform(0.35, 0.95), 2),
                max_new_tokens=rng.choice([128, 192, 256, 320]),
                top_p=round(rng.uniform(0.85, 0.98), 3) if rng.random() > 0.3 else None,
                relationships=rels,
                log_thoughts=rng.random() > 0.35,
                include_thought_phase=None if rng.random() > 0.4 else rng.choice([True, False]),
            )
        )

    events = _sample_background_events(rng, str(pick.get("events_key", "generic")), mr)

    order = turn_order or rng.choice(["round_robin", "random", "reactive"])
    thought = enable_thought_phase if enable_thought_phase is not None else (rng.random() > 0.45)
    mem = memory_turns if memory_turns is not None else rng.choice([12, 16, 20, 24])
    if sample_stop_phrase and not (stop_phrase or "").strip():
        stop = rng.choice([None, None, None, "SCENE_END", "PARLEY_NOW"]) if rng.random() > 0.7 else None
    else:
        stop = (stop_phrase or "").strip() or None

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
        memory_turns=int(mem),
        stop_phrase=stop,
        # Leave unset; engine falls back to agents[0].model_id after assign_models_to_agents.
        reactive_router_model_id=None,
    )

    return ScenarioConfig(world=world, agents=agents, orchestration=orch)
