"""Notebook files are valid nbformat v4; full execution: pytest --nbmake notebooks/01_quickstart.ipynb"""

from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks" / "01_quickstart.ipynb",
    ROOT / "notebooks" / "02_world_and_agents.ipynb",
    ROOT / "notebooks" / "03_hf_inference_backend.ipynb",
    ROOT / "notebooks" / "04_local_transformers_optional.ipynb",
    ROOT / "notebooks" / "05_reporting_and_metrics.ipynb",
    ROOT / "notebooks" / "06_full_randomized_simulation.ipynb",
    ROOT / "notebooks" / "07_simulation_exemplar.ipynb",
]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_is_valid_nbformat(path: Path):
    assert path.is_file(), f"Missing {path}"
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
