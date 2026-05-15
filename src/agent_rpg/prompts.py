from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from agent_rpg.schemas.agent import AgentConfig
from agent_rpg.schemas.scenario import ScenarioConfig


def _templates_dir() -> Path:
    return Path(__file__).resolve().parent / "templates" / "agents"


def build_agent_system_prompt(
    scenario: ScenarioConfig,
    agent: AgentConfig,
    *,
    active_events_text: str,
    round_index: int,
) -> str:
    if agent.system_prompt and not agent.prompt_template_id:
        rel = _format_relationships(scenario, agent)
        extra = (
            f"\n\n---\nRound: {round_index}\n"
            f"Active world events:\n{active_events_text or '(none)'}\n"
            f"Relationships:\n{rel}\n"
        )
        return agent.system_prompt.strip() + extra

    tmpl_id = agent.prompt_template_id or "default"
    env = Environment(
        loader=FileSystemLoader(_templates_dir()),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(f"{tmpl_id}.jinja2")
    rel = _format_relationships(scenario, agent)
    ctx = {
        "world": scenario.world,
        "agent": agent,
        "relationships_block": rel,
        "active_events_text": active_events_text,
        "round_index": round_index,
        **agent.prompt_variables,
    }
    return template.render(**ctx)


def _format_relationships(scenario: ScenarioConfig, agent: AgentConfig) -> str:
    if not agent.relationships:
        return "(none)"
    lines = []
    for other_id, desc in agent.relationships.items():
        name = other_id
        for a in scenario.agents:
            if a.agent_id == other_id:
                name = a.display_name
                break
        lines.append(f"- {name} ({other_id}): {desc}")
    return "\n".join(lines)
