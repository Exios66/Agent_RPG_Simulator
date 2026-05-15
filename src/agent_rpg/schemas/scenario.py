from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from agent_rpg.schemas.agent import AgentConfig
from agent_rpg.schemas.orchestration import OrchestrationConfig
from agent_rpg.schemas.world import WorldConfig


class ScenarioConfig(BaseModel):
    world: WorldConfig
    agents: list[AgentConfig] = Field(..., min_length=1)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)

    @model_validator(mode="after")
    def check_relationships(self) -> ScenarioConfig:
        ids = {a.agent_id for a in self.agents}
        for agent in self.agents:
            for other in agent.relationships:
                if other not in ids and other not in ("narrator", "scene", "crowd"):
                    raise ValueError(
                        f"Agent {agent.agent_id} references unknown agent_id in relationships: {other}"
                    )
        return self
