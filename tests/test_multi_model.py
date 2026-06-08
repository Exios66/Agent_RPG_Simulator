"""Tests for per-agent Hub model assignment."""

from __future__ import annotations

import random

from agent_rpg.model_catalog import DEFAULT_INSTRUCT_MODEL_ID
from agent_rpg.multi_model import assign_models_to_agents, set_router_model_if_reactive
from agent_rpg.random_scenario import build_random_scenario


def test_assign_rotate_cycles_pool():
    scenario = build_random_scenario(seed=1, num_agents=4, max_rounds=1, model_id="model-a")
    pool = ["hub/x", "hub/y"]
    assign_models_to_agents(scenario, pool, strategy="rotate", rng=random.Random(0))
    got = [a.model_id for a in scenario.agents]
    assert got == ["hub/x", "hub/y", "hub/x", "hub/y"]


def test_assign_shuffle_is_permutation_of_rotate_cycle():
    scenario = build_random_scenario(seed=2, num_agents=3, max_rounds=1, model_id="z")
    pool = ["a", "b"]
    assign_models_to_agents(scenario, pool, strategy="shuffle", rng=random.Random(99))
    got = [a.model_id for a in scenario.agents]
    assert len(got) == 3
    assert all(x in pool for x in got)


def test_assign_empty_pool_falls_back_to_default():
    scenario = build_random_scenario(seed=3, num_agents=2, max_rounds=1, model_id="custom/base")
    assign_models_to_agents(scenario, [], strategy="rotate")
    assert all(a.model_id == DEFAULT_INSTRUCT_MODEL_ID for a in scenario.agents)


def test_set_router_model_if_reactive_sets_once():
    scenario = build_random_scenario(seed=4, num_agents=2, max_rounds=1, turn_order="reactive")
    scenario.orchestration.reactive_router_model_id = None
    assign_models_to_agents(scenario, ["r1", "r2"], strategy="rotate")
    set_router_model_if_reactive(scenario, "router-model")
    assert scenario.orchestration.reactive_router_model_id == "router-model"


def test_set_router_model_if_reactive_overwrites_build_time_default():
    """After pool assignment, router must track agent 0 — not the build-time placeholder."""
    scenario = build_random_scenario(
        seed=1, num_agents=3, max_rounds=1, turn_order="reactive", model_id="build-default"
    )
    assert scenario.orchestration.reactive_router_model_id == "build-default"
    assign_models_to_agents(scenario, ["model-A", "model-B"], strategy="shuffle", rng=random.Random(99))
    set_router_model_if_reactive(scenario, scenario.agents[0].model_id)
    assert scenario.orchestration.reactive_router_model_id == scenario.agents[0].model_id
    assert scenario.orchestration.reactive_router_model_id != "build-default"
