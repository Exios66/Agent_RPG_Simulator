from __future__ import annotations

import json
import re

from pydantic import BaseModel


class ParsedTurn(BaseModel):
    thought: str = ""
    say: str = ""
    directed_at: str | None = None
    raw: str = ""
    parse_error: str | None = None


def _strip_fences(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", t, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if "```" in t:
        parts = t.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.lower().startswith("json"):
                inner = inner[4:].lstrip()
            return inner.strip()
    return t


def parse_agent_json_response(text: str) -> ParsedTurn:
    raw = text
    cleaned = _strip_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return ParsedTurn(
            say=cleaned or text,
            raw=raw,
            parse_error="invalid_json",
        )
    if not isinstance(data, dict):
        return ParsedTurn(say=str(data), raw=raw, parse_error="not_object")
    thought = str(data.get("thought", "") or "")
    say = str(data.get("say", "") or data.get("message", "") or "")
    directed = data.get("directed_at")
    directed_at = None if directed in (None, "", "null") else str(directed)
    return ParsedTurn(
        thought=thought,
        say=say,
        directed_at=directed_at,
        raw=raw,
        parse_error=None if say else "empty_say",
    )


def parse_router_response(text: str, allowed: set[str]) -> str | None:
    cleaned = _strip_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    nid = data.get("next_agent_id") or data.get("next")
    if nid is None:
        return None
    s = str(nid)
    return s if s in allowed else None
