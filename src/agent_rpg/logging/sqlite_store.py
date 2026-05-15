from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from agent_rpg.schemas.events import SimulationEvent


class SqliteEventStore:
    """Optional mirror of JSONL events for querying."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                run_id TEXT,
                round INTEGER,
                agent_id TEXT,
                event_type TEXT,
                payload TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_run_round ON events(run_id, round)"
        )
        self._conn.commit()

    def insert(self, event: SimulationEvent) -> None:
        self._conn.execute(
            "INSERT INTO events (timestamp, run_id, round, agent_id, event_type, payload) VALUES (?,?,?,?,?,?)",
            (
                event.timestamp,
                event.run_id,
                event.round,
                event.agent_id,
                event.event_type,
                json.dumps(event.payload),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SqliteEventStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
