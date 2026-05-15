from __future__ import annotations

from collections import defaultdict
from typing import Any

from agent_rpg.schemas.events import SimulationEvent


def gini(values: list[float]) -> float:
    """Gini coefficient; 0 = equal, 1 = maximally unequal."""
    if not values:
        return 0.0
    arr = sorted(float(x) for x in values)
    n = len(arr)
    if n == 1:
        return 0.0
    cum = 0.0
    for i, v in enumerate(arr, start=1):
        cum += (2 * i - n - 1) * v
    denom = n * sum(arr)
    return cum / denom if denom else 0.0


def compute_social_metrics(events: list[SimulationEvent]) -> dict[str, Any]:
    msgs = [e for e in events if e.event_type == "message" and e.agent_id]
    turns_per_agent: dict[str, int] = defaultdict(int)
    chars_per_agent: dict[str, int] = defaultdict(int)
    directed: dict[tuple[str, str | None], int] = defaultdict(int)
    pairwise: dict[tuple[str, str], int] = defaultdict(int)
    decisions: list[dict[str, Any]] = []

    for e in msgs:
        aid = e.agent_id or ""
        text = str(e.payload.get("text", ""))
        turns_per_agent[aid] += 1
        chars_per_agent[aid] += len(text)
        to = e.payload.get("directed_at")
        to_key: str | None = str(to) if to else None
        directed[(aid, to_key)] += 1
        if to_key:
            pairwise[(aid, to_key)] += 1
        if "DECISION:" in text.upper() or "decision:" in text:
            decisions.append({"round": e.round, "agent_id": aid, "excerpt": text[:500]})

    tp = list(turns_per_agent.values())
    reciprocity: dict[str, int] = {}
    agent_ids = sorted(turns_per_agent.keys())
    for i, a in enumerate(agent_ids):
        for b in agent_ids[i + 1 :]:
            reciprocity[f"{a}<->{b}"] = min(pairwise.get((a, b), 0), pairwise.get((b, a), 0))

    return {
        "turn_counts": dict(turns_per_agent),
        "char_counts": dict(chars_per_agent),
        "gini_turns": gini([float(x) for x in tp]) if tp else 0.0,
        "directed_edges": {f"{a}->{b}": c for (a, b), c in directed.items()},
        "pairwise_directed": {f"{a}->{b}": c for (a, b), c in pairwise.items()},
        "reciprocity_min_counts": reciprocity,
        "decisions": decisions,
    }
