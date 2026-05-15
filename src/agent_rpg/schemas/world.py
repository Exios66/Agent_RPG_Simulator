from __future__ import annotations

from pydantic import BaseModel, Field


class BackgroundEvent(BaseModel):
    """World event optionally tied to a round."""

    id: str = Field(..., description="Stable id for references")
    description: str
    round_trigger: int | None = Field(
        default=None,
        ge=0,
        description="If set, event is active starting this round index (0-based)",
    )


class WorldConfig(BaseModel):
    scenario_id: str
    title: str
    version: str = "1"
    setting: str = ""
    backstory: str = ""
    worldbuilding_notes: str = ""
    background_events: list[BackgroundEvent] = Field(default_factory=list)
    max_rounds: int = Field(default=5, ge=1, le=10_000)
    round_duration_narrative: str = ""
    default_occupations: list[str] = Field(default_factory=list)
