from pathlib import Path

import pytest

from agent_rpg.repo_root import find_repo_root


def test_find_repo_root_from_here(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Simulate cwd inside a subfolder of the real repo
    real = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(real / "tests")
    root = find_repo_root()
    assert (root / "examples" / "scenarios").is_dir()


def test_find_repo_root_env_override(monkeypatch: pytest.MonkeyPatch):
    real = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("AGENT_RPG_ROOT", str(real))
    monkeypatch.chdir("/")
    assert find_repo_root() == real
