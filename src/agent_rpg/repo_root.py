"""Locate the repository root (contains ``examples/scenarios``) for notebooks and scripts."""

from __future__ import annotations

import os
from pathlib import Path


def find_repo_root() -> Path:
    """
    Resolve the Agent RPG Simulator repo root.

    Order:
    1. Environment variable ``AGENT_RPG_ROOT``
    2. Walk upward from :func:`Path.cwd` looking for ``pyproject.toml`` + ``examples/scenarios``
    3. Infer from installed ``agent_rpg`` package path (editable installs)
    """
    env = os.environ.get("AGENT_RPG_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if not (p / "examples" / "scenarios").is_dir():
            raise FileNotFoundError(
                f"AGENT_RPG_ROOT={p!s} does not contain examples/scenarios/"
            )
        return p

    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        if (base / "pyproject.toml").is_file() and (base / "examples" / "scenarios").is_dir():
            return base

    try:
        import agent_rpg

        pkg_root = Path(agent_rpg.__file__).resolve().parents[2]
        if (pkg_root / "examples" / "scenarios").is_dir():
            return pkg_root
    except ImportError:
        pass

    raise FileNotFoundError(
        "Could not locate repository root (expected pyproject.toml and examples/scenarios/). "
        "Start Jupyter from the repo root, `cd` there before running cells, or set AGENT_RPG_ROOT."
    )
