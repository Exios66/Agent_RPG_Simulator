from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

from agent_rpg.schemas.events import SimulationEvent


class JsonlEventWriter:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp: TextIO = self.path.open("w", encoding="utf-8")

    def write(self, event: SimulationEvent) -> None:
        self._fp.write(event.model_dump_json() + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()

    def __enter__(self) -> JsonlEventWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def iter_events_jsonl(path: Path | str) -> list[SimulationEvent]:
    p = Path(path)
    out: list[SimulationEvent] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(SimulationEvent.model_validate_json(line))
    return out
