# Agent RPG Simulator

Python package for **multi-agent LLM roleplay simulations**: YAML-defined worlds and agents, pluggable backends (Hugging Face Inference API by default, optional local `transformers`), JSONL event logs, optional SQLite mirror, and reporting helpers.

## Setup

Use Python 3.10+ (3.11 recommended).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

### Pip `requirements*.txt` (alternative to extras)

| File | Use case |
|------|-----------|
| [`requirements.txt`](requirements.txt) | Minimal: remote HF Inference + simulator core |
| [`requirements-dev.txt`](requirements-dev.txt) | Tests, coverage, nbmake / nbformat |
| [`requirements-notebooks.txt`](requirements-notebooks.txt) | Jupyter + matplotlib + dotenv |
| [`requirements-local.txt`](requirements-local.txt) | Local `transformers` + `torch` |
| [`requirements-all.txt`](requirements-all.txt) | Everything above |

Typical flows: `pip install -r requirements.txt && pip install -e .` then add `-r requirements-dev.txt` for development, or `-r requirements-notebooks.txt` before running notebooks.

Optional extras (same tiers via pyproject):

- **Local GPU/CPU models:** `pip install -e '.[local]'`
- **Notebooks:** `pip install -e '.[notebooks]'`

## Hugging Face token

Set `HF_TOKEN` in your environment for remote inference (never commit tokens). For local notebooks you can also put the token on the **first line** of **`hf_token.txt`** at the repo root (that filename is gitignored).

- Copy [`.env.example`](.env.example) to `.env` and set `HF_TOKEN` there. The `agent-rpg` CLI loads `.env` automatically when `python-dotenv` is installed (`requirements-notebooks.txt` / `pip install -e '.[notebooks]'`).
- In notebooks, call `load_dotenv()` as in the template below.

```python
from dotenv import load_dotenv
load_dotenv()
```

### Inference credits (HTTP 402)

Serverless **Inference Providers** (the default `InferenceClient` router) are billed against your Hugging Face account. If you see **HTTP 402 Payment Required**, included credits are exhausted until you add prepaid credits, upgrade (e.g. PRO), or reduce usage. For development without the paid router, use **`FakeLLMBackend`** in notebooks, **`pip install -e '.[local]'`** with **`TransformersLocalBackend`**, or fewer agents/rounds/smaller `max_new_tokens`.

## Run a scenario

```bash
export HF_TOKEN=hf_...
agent-rpg run examples/scenarios/tavern.yaml --output runs
```

Flags:

- `--backend hf` (default) or `--backend local` for `TransformersLocalBackend` on this machine.
- `--save-sqlite` writes `runs/<run-id>/events.sqlite`.
- `--sqlite /path/to.db` for an explicit SQLite path.
- `--run-id my_run` and `--seed 42` for reproducibility (random turn order).

Agents with `backend: transformers_local` in YAML require `SimulationEngine.run(..., local_backend=TransformersLocalBackend())` (or use CLI `--backend local` with a scenario that only uses local-compatible settings).

## Library usage

```python
from agent_rpg import (
    SimulationEngine,
    load_scenario,
    ReportBuilder,
    scenario_json_schema,
    build_random_scenario,
    find_repo_root,
)
from agent_rpg.backends.fake import FakeLLMBackend
from agent_rpg.backends.hf_inference import HuggingFaceInferenceBackend

scenario = load_scenario("examples/scenarios/minimal.yaml")
engine = SimulationEngine(scenario)
out = engine.run(HuggingFaceInferenceBackend(), output_dir="runs")  # or FakeLLMBackend(...)
report = ReportBuilder.from_jsonl(out / "events.jsonl")
report.write_markdown(out / "report.md", scenario=scenario)
```

`scenario_json_schema()` returns the JSON Schema for `ScenarioConfig` (for editors and tooling).

**Multi-model runs:** after building or loading a scenario, call `assign_models_to_agents(scenario, model_pool, strategy="rotate"|"shuffle"|"random")` so each agent’s YAML `model_id` is overwritten from the pool. For `turn_order: reactive`, set `orchestration.reactive_router_model_id` or use `set_router_model_if_reactive(scenario, pool[0])`. Defaults and a curated small-instruct list live in `agent_rpg.model_catalog` (default Hub id: **`meta-llama/Llama-3.1-8B-Instruct`**).

Procedural scenarios: `build_random_scenario(seed=...)` returns a new valid `ScenarioConfig` each time (see `notebooks/06_full_randomized_simulation.ipynb` and the full tour in `notebooks/07_simulation_exemplar.ipynb`).

## Notebooks

Under `notebooks/`:

| File | Purpose |
|------|---------|
| `01_quickstart.ipynb` | Mock backend run and JSONL tail |
| `02_world_and_agents.ipynb` | Load/validate YAML |
| `03_hf_inference_backend.ipynb` | HF Inference when `HF_TOKEN` is set; mock otherwise |
| `04_local_transformers_optional.ipynb` | Optional `[local]` backend |
| `05_reporting_and_metrics.ipynb` | Reports and simple plots |
| `06_full_randomized_simulation.ipynb` | **Procedural** world/agents/events/orchestration + full mock (or HF) run + report |
| `07_simulation_exemplar.ipynb` | **Full walkthrough** of every simulation element + **`SIMULATION_CONFIG`** tailored randomized run (incl. optional **`model_pool`**) |
| `08_live_hf_spectator.ipynb` | **Live HF** multi-model run with **ipywidgets** model pool + **`on_event`** real-time spectator feed |

**Recommended:** open **`07_simulation_exemplar.ipynb`** first for the complete narrative; **`06`** is a shorter env-var-driven variant. For **live** Hub inference with per-agent models and a streaming-style transcript, use **`08_live_hf_spectator.ipynb`** (requires `HF_TOKEN` and Hub access to the selected models).

Execute notebooks end-to-end (requires a working Jupyter kernel):

```bash
pip install -e '.[notebooks]'
pytest --nbmake notebooks/01_quickstart.ipynb notebooks/02_world_and_agents.ipynb notebooks/05_reporting_and_metrics.ipynb notebooks/06_full_randomized_simulation.ipynb notebooks/07_simulation_exemplar.ipynb
```

Regenerate all notebooks (including `07` from `scripts/exemplar_07.py`): `python scripts/rebuild_notebooks.py`.

## Tests

```bash
pytest tests/
```

Live Hugging Face smoke test (optional):

```bash
export RUN_HF_LIVE=1
export HF_TOKEN=...
# optional: HF_LIVE_MODEL=meta-llama/Llama-3.1-8B-Instruct  (defaults to same via model_catalog)
pytest tests/test_hf_live.py -m live_hf
```

## Example scenarios

See `examples/scenarios/`: `minimal.yaml`, `tavern.yaml`, `council.yaml`, `border_conflict.yaml`.
