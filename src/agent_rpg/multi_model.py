"""Assign different Hub `model_id` values per agent for multi-model community runs."""

from __future__ import annotations

import random
from typing import Literal, Sequence

from agent_rpg.model_catalog import DEFAULT_INSTRUCT_MODEL_ID
from agent_rpg.schemas.scenario import ScenarioConfig


def assign_models_to_agents(
    scenario: ScenarioConfig,
    models: Sequence[str],
    *,
    strategy: Literal["rotate", "shuffle", "random"] = "rotate",
    rng: random.Random | None = None,
) -> ScenarioConfig:
    """
    Mutate each agent's ``model_id`` from a pool of Hub repo ids.

    - ``rotate``: agent i gets ``models[i % len(models)]``
    - ``shuffle``: cycle order is shuffled once, then rotate
    - ``random``: each agent picks uniformly from pool (with optional ``rng``)
    """
    pool = [m for m in models if m.strip()]
    if not pool:
        pool = [DEFAULT_INSTRUCT_MODEL_ID]
    r = rng or random.Random()
    order = list(pool)
    if strategy == "shuffle":
        r.shuffle(order)
    for i, agent in enumerate(scenario.agents):
        if strategy == "random":
            agent.model_id = r.choice(order)
        else:
            agent.model_id = order[i % len(order)]
    return scenario


def set_router_model_if_reactive(scenario: ScenarioConfig, model_id: str | None = None) -> None:
    """If turn order is reactive, set the router model.

    When ``model_id`` is passed explicitly (e.g. after ``assign_models_to_agents``), always
    apply it so a stale value from scenario construction does not stick. When omitted, only fill
    in if still unset.
    """
    if scenario.orchestration.turn_order != "reactive":
        return
    if model_id is not None:
        scenario.orchestration.reactive_router_model_id = model_id
        return
    if scenario.orchestration.reactive_router_model_id is not None:
        return
    scenario.orchestration.reactive_router_model_id = (
        scenario.agents[0].model_id if scenario.agents else DEFAULT_INSTRUCT_MODEL_ID
    )
