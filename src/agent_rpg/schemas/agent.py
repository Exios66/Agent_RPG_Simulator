from __future__ import annotations

from typing import Any, Literal

from agent_rpg.model_catalog import DEFAULT_INSTRUCT_MODEL_ID
from pydantic import BaseModel, Field

BackendName = Literal["auto", "hf_inference", "transformers_local", "openrouter"]


class AgentConfig(BaseModel):
    agent_id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$")
    display_name: str
    archetype: str = ""
    occupation: str = ""
    system_prompt: str | None = None
    prompt_template_id: str | None = None
    prompt_variables: dict[str, Any] = Field(default_factory=dict)
    model_id: str = Field(
        default=DEFAULT_INSTRUCT_MODEL_ID,
        description=(
            "Model id for the agent's backend: Hub repo id for HF Inference / auto; local path or Hub id for "
            "transformers_local; OpenRouter model slug (e.g. …:free) when backend is openrouter. "
            "Use multi_model.assign_models_to_agents for HF pools."
        ),
    )
    backend: BackendName = "auto"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=256, ge=1, le=8192)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    relationships: dict[str, str] = Field(
        default_factory=dict,
        description="Map other agent_id -> relationship description",
    )
    log_thoughts: bool = Field(default=True)
    include_thought_phase: bool | None = Field(
        default=None,
        description="Override orchestration thought phase; None = use global",
    )
    turn_weight: float = Field(default=1.0, ge=0.0)
