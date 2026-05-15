from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_rpg.logging.jsonl import iter_events_jsonl
from agent_rpg.reporting.metrics import compute_social_metrics
from agent_rpg.schemas.events import SimulationEvent
from agent_rpg.schemas.scenario import ScenarioConfig


class ReportBuilder:
    """Build structured summaries from a finished JSONL run."""

    def __init__(self, events: list[SimulationEvent]) -> None:
        self.events = events

    @classmethod
    def from_jsonl(cls, path: str | Path) -> ReportBuilder:
        return cls(iter_events_jsonl(path))

    def to_dict(self, scenario: ScenarioConfig | None = None) -> dict[str, Any]:
        by_round: dict[int, list[SimulationEvent]] = {}
        for e in self.events:
            by_round.setdefault(e.round, []).append(e)

        thoughts = [e for e in self.events if e.event_type == "thought"]
        messages = [e for e in self.events if e.event_type == "message"]
        errors = [e for e in self.events if e.event_type == "error"]
        world_events = [e for e in self.events if e.event_type == "world_event"]

        per_agent: dict[str, dict[str, Any]] = {}
        for e in messages:
            aid = e.agent_id or "unknown"
            per_agent.setdefault(aid, {"quotes": [], "turns": 0})
            per_agent[aid]["turns"] += 1
            text = str(e.payload.get("text", ""))
            if len(text) > 120:
                per_agent[aid]["quotes"].append({"round": e.round, "text": text[:400]})

        social = compute_social_metrics(self.events)

        relationships_config: dict[str, dict[str, str]] | None = None
        if scenario is not None:
            relationships_config = {
                a.agent_id: dict(a.relationships) for a in scenario.agents
            }

        out: dict[str, Any] = {
            "summary": {
                "total_events": len(self.events),
                "rounds_observed": sorted(by_round.keys()),
                "thought_count": len(thoughts),
                "message_count": len(messages),
                "error_count": len(errors),
                "world_event_count": len(world_events),
            },
            "timeline": {
                str(r): [ev.model_dump() for ev in evs]
                for r, evs in sorted(by_round.items(), key=lambda x: x[0])
            },
            "per_agent": per_agent,
            "social_dynamics": social,
        }
        if relationships_config is not None:
            out["relationships_config"] = relationships_config
        return out

    def write_markdown(self, path: str | Path, scenario: ScenarioConfig | None = None) -> None:
        d = self.to_dict(scenario=scenario)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Simulation report", ""]
        s = d["summary"]
        lines.append(f"- Total events: {s['total_events']}")
        lines.append(f"- Messages: {s['message_count']}, Thoughts: {s['thought_count']}, Errors: {s['error_count']}")
        lines.append("")
        lines.append("## Social dynamics")
        lines.append("```json")
        lines.append(json.dumps(d["social_dynamics"], indent=2))
        lines.append("```")
        lines.append("")
        if "relationships_config" in d:
            lines.append("## Configured relationships (scenario)")
            lines.append("```json")
            lines.append(json.dumps(d["relationships_config"], indent=2))
            lines.append("```")
            lines.append("")
        lines.append("## Per-agent highlights")
        for aid, block in d["per_agent"].items():
            lines.append(f"### {aid}")
            lines.append(f"- Turns: {block['turns']}")
            for q in block.get("quotes", [])[:3]:
                lines.append(f"- Round {q['round']}: _{q['text'][:200]}..._")
            lines.append("")
        p.write_text("\n".join(lines), encoding="utf-8")
