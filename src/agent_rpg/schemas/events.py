from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

EventType = Literal[
    "thought",
    "message",
    "system",
    "world_event",
    "error",
    "metric",
    "router",
]


class SimulationEvent(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str
    round: int = Field(ge=0)
    agent_id: str | None = None
    event_type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
