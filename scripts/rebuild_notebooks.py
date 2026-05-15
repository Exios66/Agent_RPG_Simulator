#!/usr/bin/env python3
"""Regenerate Jupyter notebooks under notebooks/ (run from repo root)."""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

BOOT = """from pathlib import Path
import sys
for base in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
    src = base / "src"
    if (src / "agent_rpg").is_dir():
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        break
from agent_rpg.repo_root import find_repo_root
ROOT = find_repo_root()
print("Repository root:", ROOT)
"""


def _id() -> str:
    return str(uuid.uuid4())[:8]


def _write(name: str, cells: list) -> None:
    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    for c in nb.cells:
        c.setdefault("id", _id())
    p = Path(__file__).resolve().parents[1] / "notebooks" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, p)
    print("Wrote", p)


def main() -> None:
    _write(
        "01_quickstart.ipynb",
        [
            new_markdown_cell(
                "# 01 Quickstart\n"
                "Load a scenario, run with a **mock** backend, inspect JSONL output.\n"
                "\n"
                "Paths use `find_repo_root()` so this works when the Jupyter cwd is `notebooks/` or the repo root."
            ),
            new_code_cell(BOOT),
            new_code_cell(
                "from agent_rpg import SimulationEngine, load_scenario\n"
                "from agent_rpg.backends.fake import FakeLLMBackend\n"
                "\n"
                "scenario = load_scenario(ROOT / \"examples\" / \"scenarios\" / \"minimal.yaml\")\n"
                "scenario.orchestration.enable_thought_phase = False\n"
                "scenario.orchestration.max_rounds = 2\n"
                "scenario.world.max_rounds = 2\n"
                "\n"
                "def fac(i, msgs):\n"
                '    return \'{\"thought\":\"\",\"say\":\"Hello from mock\",\"directed_at\":null}\'\n'
                "\n"
                "backend = FakeLLMBackend(factory=fac)\n"
                'out = SimulationEngine(scenario).run(backend, output_dir=ROOT / "runs", run_id="nb01_demo")\n'
                'print("Wrote:", out / "events.jsonl")'
            ),
            new_code_cell(
                'lines = (out / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()\n'
                'print("Events:", len(lines))\n'
                "print(lines[-1][:240], \"...\")"
            ),
        ],
    )

    _write(
        "02_world_and_agents.ipynb",
        [
            new_markdown_cell(
                "# 02 World and agents\n"
                "Load example YAML, validate with Pydantic, and show a controlled validation failure."
            ),
            new_code_cell(BOOT),
            new_code_cell(
                "from pydantic import ValidationError\n"
                "from agent_rpg.loader import load_scenario\n"
                "from agent_rpg.schemas.scenario import ScenarioConfig\n"
                "\n"
                's = load_scenario(ROOT / "examples" / "scenarios" / "tavern.yaml")\n'
                'print(s.world.title, "—", len(s.agents), "agents")'
            ),
            new_code_cell(
                "bad = {\n"
                '    "world": {"scenario_id": "x", "title": "t", "max_rounds": 1},\n'
                '    "agents": [{"agent_id": "a", "display_name": "A", "relationships": {"nobody": "x"}}],\n'
                "}\n"
                "try:\n"
                "    ScenarioConfig.model_validate(bad)\n"
                "except ValidationError:\n"
                '    print("Caught ValidationError as expected")'
            ),
        ],
    )

    _write(
        "03_hf_inference_backend.ipynb",
        [
            new_markdown_cell(
                "# 03 Hugging Face Inference\n"
                "Uses **`HF_TOKEN`** for remote inference (cost / rate limits apply).\n"
                "\n"
                "If `HF_TOKEN` is unset, this notebook uses `FakeLLMBackend` so cells still run."
            ),
            new_code_cell(BOOT + "import os\n"),
            new_code_cell(
                "from agent_rpg import SimulationEngine, load_scenario\n"
                "from agent_rpg.backends.fake import FakeLLMBackend\n"
                "from agent_rpg.backends.hf_inference import HuggingFaceInferenceBackend\n"
                "\n"
                'scenario = load_scenario(ROOT / "examples" / "scenarios" / "minimal.yaml")\n'
                "scenario.orchestration.max_rounds = 1\n"
                "scenario.world.max_rounds = 1\n"
                "scenario.orchestration.enable_thought_phase = False\n"
                "\n"
                "if os.environ.get(\"HF_TOKEN\"):\n"
                "    backend = HuggingFaceInferenceBackend()\n"
                '    print("Using HuggingFaceInferenceBackend")\n'
                "else:\n"
                "    backend = FakeLLMBackend(\n"
                "        factory=lambda i, m: '{\\\"thought\\\":\\\"\\\",\\\"say\\\":\\\"no token demo\\\",\\\"directed_at\\\":null}',\n"
                "    )\n"
                '    print("HF_TOKEN not set; using FakeLLMBackend")\n'
                "\n"
                'out = SimulationEngine(scenario).run(backend, output_dir=ROOT / "runs", run_id="nb03_hf")\n'
                "print(out)"
            ),
        ],
    )

    _write(
        "04_local_transformers_optional.ipynb",
        [
            new_markdown_cell(
                "# 04 Local Transformers (optional)\n"
                "Requires `pip install 'agent-rpg[local]'` (transformers + torch). "
                "If the import fails, install the extra and restart the kernel."
            ),
            new_code_cell(BOOT),
            new_code_cell(
                "try:\n"
                "    from agent_rpg.backends.transformers_local import TransformersLocalBackend\n"
                '    print("TransformersLocalBackend:", TransformersLocalBackend)\n'
                "except ImportError as e:\n"
                '    print("Optional local backend not installed:", e)'
            ),
            new_code_cell(
                'print("CLI: agent-rpg run scenario.yaml --backend local  (after installing [local])")'
            ),
        ],
    )

    _write(
        "05_reporting_and_metrics.ipynb",
        [
            new_markdown_cell(
                "# 05 Reporting and metrics\n"
                "Run a short mock simulation, build a `ReportBuilder`, and plot turn counts."
            ),
            new_code_cell(BOOT),
            new_code_cell(
                "from agent_rpg import ReportBuilder, SimulationEngine, load_scenario\n"
                "from agent_rpg.backends.fake import FakeLLMBackend\n"
                "\n"
                'scenario = load_scenario(ROOT / "examples" / "scenarios" / "minimal.yaml")\n'
                "scenario.orchestration.enable_thought_phase = False\n"
                "scenario.orchestration.max_rounds = 1\n"
                "scenario.world.max_rounds = 1\n"
                "out = SimulationEngine(scenario).run(\n"
                "    FakeLLMBackend(\n"
                "        factory=lambda i, m: '{\\\"thought\\\":\\\"\\\",\\\"say\\\":\\\"hello there\\\",\\\"directed_at\\\":\\\"bob\\\"}',\n"
                "    ),\n"
                '    output_dir=ROOT / "runs",\n'
                '    run_id="nb05_metrics",\n'
                ")\n"
                'rb = ReportBuilder.from_jsonl(out / "events.jsonl")\n'
                "d = rb.to_dict(scenario=scenario)\n"
                'print(d["summary"])\n'
                'print(d["social_dynamics"]["turn_counts"])'
            ),
            new_code_cell(
                "try:\n"
                "    import matplotlib.pyplot as plt\n"
                "\n"
                '    counts = d["social_dynamics"]["turn_counts"]\n'
                "    fig, ax = plt.subplots(figsize=(6, 3))\n"
                "    ax.bar(list(counts.keys()), list(counts.values()))\n"
                '    ax.set_title("Messages per agent")\n'
                "    plt.show()\n"
                "except Exception as e:\n"
                '    print("Plot skipped:", e)'
            ),
        ],
    )

    nb06_run = """
import os
import random

from agent_rpg import build_random_scenario

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

SEED = int(os.environ.get("NB_RANDOM_SEED", random.randint(1, 10_000_000)))
USE_HF = os.environ.get("NB_USE_HF", "").lower() in ("1", "true", "yes")
NUM_AGENTS = int(os.environ.get("NB_NUM_AGENTS", "0")) or None
MAX_ROUNDS = int(os.environ.get("NB_MAX_ROUNDS", "0")) or None

scenario = build_random_scenario(seed=SEED, num_agents=NUM_AGENTS, max_rounds=MAX_ROUNDS)
scenario.orchestration.enable_thought_phase = False
for a in scenario.agents:
    a.max_new_tokens = min(a.max_new_tokens, 180)

print("SEED", SEED)
print("Title:", scenario.world.title)
print("Agents:", [a.display_name for a in scenario.agents])
print("Turn order:", scenario.orchestration.turn_order)
print("World events:", len(scenario.world.background_events))
""".strip()

    nb06_events = """
_lines = []
for ev in scenario.world.background_events:
    rt = ev.round_trigger
    _lines.append(f"- [{ev.id}] (from round {rt}) {ev.description}")
print("\\n".join(_lines) if _lines else "(no scripted world events)")
""".strip()

    nb06_backend = """
import json
import os
import random

from agent_rpg import SimulationEngine
from agent_rpg.backends.fake import FakeLLMBackend
from agent_rpg.backends.hf_inference import HuggingFaceInferenceBackend

if USE_HF and os.environ.get("HF_TOKEN"):
    backend = HuggingFaceInferenceBackend()
    print("Backend: HuggingFaceInferenceBackend")
else:
    if USE_HF and not os.environ.get("HF_TOKEN"):
        print("USE_HF requested but HF_TOKEN missing — falling back to mock.")

    first_id = scenario.agents[0].agent_id

    def fake_factory(i, msgs):
        sysm = (msgs[0].get("content") or "") if msgs else ""
        if "scene director" in sysm.lower():
            return json.dumps({"next_agent_id": first_id})
        sp = (scenario.orchestration.stop_phrase or "") or ""
        opts = [
            {"thought": "weigh options", "say": "I propose we compare accounts.", "directed_at": None},
            {"thought": "pressure rising", "say": "Name your price, plainly.", "directed_at": None},
            {"thought": "", "say": "Let us not waste the night on insults.", "directed_at": None},
        ]
        r = random.Random(SEED + i)
        row = dict(opts[r.randint(0, len(opts) - 1)])
        if sp and r.random() < 0.25:
            row = {"thought": "endgame", "say": sp, "directed_at": None}
        return json.dumps(row)

    backend = FakeLLMBackend(factory=fake_factory)
    print("Backend: FakeLLMBackend (set NB_USE_HF=1 and HF_TOKEN for live)")

run_id = f"nb06_rand_{SEED}"
engine = SimulationEngine(scenario)
out = engine.run(backend, output_dir=ROOT / "runs", run_id=run_id, seed=SEED)
print("Artifacts:", out)
""".strip()

    nb06_report = """
from agent_rpg import ReportBuilder

rb = ReportBuilder.from_jsonl(out / "events.jsonl")
report_path = out / "report.md"
rb.write_markdown(report_path, scenario=scenario)
full = rb.to_dict(scenario=scenario)
summary = full["summary"]
social = full["social_dynamics"]
print("Report:", report_path)
print("Messages:", summary["message_count"], "Errors:", summary["error_count"])
print("Gini(turns):", round(social["gini_turns"], 4))
""".strip()

    nb06_plot = """
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    tc = social["turn_counts"]
    axes[0].bar(list(tc.keys()), list(tc.values()))
    axes[0].set_title("Messages per agent")
    de = {k: v for k, v in social.get("directed_edges", {}).items() if v and "->" in k}
    if de:
        axes[1].bar(list(de.keys()), list(de.values()))
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].set_title("Directed edges (counts)")
    else:
        axes[1].set_visible(False)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Plot skipped:", e)
""".strip()

    _write(
        "06_full_randomized_simulation.ipynb",
        [
            new_markdown_cell(
                "# 06 Full randomized simulation (compact)\n"
                "Each run samples **world premise**, **background events**, **agent roster**, "
                "**relationships**, **orchestration**, and **hyperparameters**.\n"
                "\n"
                "For a **guided tour of every subsystem** plus the same style of randomized run with a "
                "**`SIMULATION_CONFIG`** dict, open **`07_simulation_exemplar.ipynb`**.\n"
                "\n"
                "- **Mock** (default): set `NB_RANDOM_SEED` for a fixed draw.\n"
                "- **HF**: set `HF_TOKEN` in `.env`, then `NB_USE_HF=1`.\n"
                "\n"
                "Optional: `pip install ipywidgets` silences tqdm widget warnings in some Jupyter setups."
            ),
            new_code_cell(BOOT),
            new_code_cell(nb06_run),
            new_code_cell(nb06_events),
            new_code_cell(nb06_backend),
            new_code_cell(nb06_report),
            new_code_cell(nb06_plot),
        ],
    )

    exemplar_path = Path(__file__).resolve().parent / "exemplar_07.py"
    spec = importlib.util.spec_from_file_location("exemplar_07", exemplar_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    _write("07_simulation_exemplar.ipynb", mod.build_cells(BOOT))

    spectator_path = Path(__file__).resolve().parent / "spectator_08.py"
    spec8 = importlib.util.spec_from_file_location("spectator_08", spectator_path)
    mod8 = importlib.util.module_from_spec(spec8)
    assert spec8.loader is not None
    spec8.loader.exec_module(mod8)
    _write("08_live_hf_spectator.ipynb", mod8.build_cells(BOOT))


if __name__ == "__main__":
    main()
