from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TurnOrder = Literal["round_robin", "random", "reactive"]


class OrchestrationConfig(BaseModel):
    turn_order: TurnOrder = "round_robin"
    max_rounds: int | None = Field(
        default=None,
        ge=1,
        description="If set, overrides world.max_rounds",
    )
    enable_thought_phase: bool = False
    memory_turns: int = Field(
        default=20,
        ge=0,
        le=500,
        description="How many prior transcript lines (agent messages) are included in conversation context",
    )
    stop_phrase: str | None = Field(
        default=None,
        description="If any public message contains this phrase (case-insensitive), end simulation",
    )
    reactive_router_model_id: str | None = Field(
        default=None,
        description="Model id for reactive turn order; defaults to first agent's model_id",
    )
