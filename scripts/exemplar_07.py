"""Cell factory for ``07_simulation_exemplar.ipynb`` (built by ``rebuild_notebooks.py``)."""

from __future__ import annotations

from nbformat.v4 import new_code_cell, new_markdown_cell


def build_cells(boot: str) -> list:
    """Return nbformat cells: full walkthrough + configurable randomized run."""
    return [
        new_markdown_cell(
            "# Simulation exemplar: every moving part\n"
            "\n"
            "This notebook **explains and demonstrates** the Agent RPG pipeline end-to-end:\n"
            "\n"
            "1. **Repository paths** — why `find_repo_root()` matters in Jupyter  \n"
            "2. **Scenario schema** — `WorldConfig`, `AgentConfig`, `OrchestrationConfig`  \n"
            "3. **World** — setting, backstory, `background_events`, pacing  \n"
            "4. **Agents** — identity, Jinja prompts, Hub `model_id`, relationships  \n"
            "5. **Orchestration** — turn order, memory window, thoughts, stop phrase  \n"
            "6. **Structured LLM output** — JSON `thought` / `say` / `directed_at`  \n"
            "7. **Backends** — mock vs Hugging Face Inference vs (optional) local  \n"
            "8. **`SimulationEngine`** — rounds, speakers, logging  \n"
            "9. **JSONL events** — `event_type` timeline  \n"
            "10. **`ReportBuilder`** — summaries and social metrics  \n"
            "11. **`SIMULATION_CONFIG`** — your knobs for the procedural run  \n"
            "12. **`build_random_scenario`** — randomized world + cast + dynamics  \n"
            "13. **Execute, report, plot** — full tailored pipeline  \n"
            "\n"
            "> Edit **`SIMULATION_CONFIG`** (Step 11), then run Steps 12–13 (and the plot cell)."
        ),
        new_code_cell(boot),
        # --- Step 1 ---
        new_markdown_cell(
            "## Step 1 — Repository root (`find_repo_root`)\n"
            "Notebooks often start with the working directory on `notebooks/`, so relative paths "
            "like `examples/scenarios/...` would break. `find_repo_root()` walks upward until it "
            "sees `pyproject.toml` and `examples/scenarios/`. You can override with env var **`AGENT_RPG_ROOT`**."
        ),
        new_code_cell(
            "from pathlib import Path\n"
            "\n"
            "examples = ROOT / \"examples\" / \"scenarios\"\n"
            "assert examples.is_dir(), examples\n"
            "print(\"Scenario YAML files:\")\n"
            "for p in sorted(examples.glob(\"*.yaml\")):\n"
            "    print(\" -\", p.name)"
        ),
        # --- Step 2 ---
        new_markdown_cell(
            "## Step 2 — Load a typed scenario (`load_scenario`)\n"
            "Scenarios are **YAML → Pydantic**: invalid relationships or missing fields fail fast. "
            "We load the small **`minimal.yaml`** for a stable reference, then later build a random world."
        ),
        new_code_cell(
            "from agent_rpg.loader import load_scenario, scenario_json_schema\n"
            "\n"
            "ref = load_scenario(ROOT / \"examples\" / \"scenarios\" / \"minimal.yaml\")\n"
            "schema = scenario_json_schema()\n"
            "print(\"Top-level schema keys:\", sorted(schema.get(\"properties\", {}).keys()))\n"
            "print(\"Loaded scenario_id:\", ref.world.scenario_id, \"| agents:\", len(ref.agents))"
        ),
        # --- Step 3 ---
        new_markdown_cell(
            "## Step 3 — World layer (`WorldConfig`)\n"
            "- **`setting` / `backstory` / `worldbuilding_notes`** — injected into agent system prompts  \n"
            "- **`background_events`** — optional `round_trigger` (event active from that round onward)  \n"
            "- **`max_rounds`** — hard cap on simulation length (orchestration can override)"
        ),
        new_code_cell(
            "w = ref.world\n"
            "print(\"Title:\", w.title)\n"
            "print(\"max_rounds:\", w.max_rounds)\n"
            "print(\"background_events:\", len(w.background_events))\n"
            "for ev in w.background_events:\n"
            "    snip = ev.description if len(ev.description) <= 72 else ev.description[:72] + \"...\"\n"
            "    print(f\"  - [{ev.id}] trigger={ev.round_trigger!r} :: {snip}\")\n"
            "if not w.background_events:\n"
            '    print("  (none in minimal.yaml)")'
        ),
        # --- Step 4 ---
        new_markdown_cell(
            "## Step 4 — Agents (`AgentConfig`)\n"
            "Each agent has **`agent_id`**, **`display_name`**, **`archetype`**, **`occupation`**, "
            "**`model_id`** (Hub repo for Inference), **`temperature` / `max_new_tokens`**, "
            "**`relationships`** (map of other `agent_id` → prose), and either **`system_prompt`** "
            "or **`prompt_template_id`** (Jinja under `src/agent_rpg/templates/agents/`)."
        ),
        new_code_cell(
            "for a in ref.agents:\n"
            "    print(f\"{a.agent_id}: {a.display_name} | {a.archetype} | model={a.model_id}\")\n"
            "    print(\"   relationships:\", a.relationships)"
        ),
        # --- Step 5 ---
        new_markdown_cell(
            "## Step 5 — Orchestration (`OrchestrationConfig`)\n"
            "- **`turn_order`**: `round_robin` | `random` | `reactive` (extra router LLM call)  \n"
            "- **`memory_turns`** — how many prior transcript lines inform context  \n"
            "- **`enable_thought_phase`** — log `thought` events when the model fills `thought`  \n"
            "- **`stop_phrase`** — end early if any `say` contains this substring (case-insensitive)"
        ),
        new_code_cell(
            "o = ref.orchestration\n"
            "print(\"turn_order:\", o.turn_order)\n"
            "print(\"memory_turns:\", o.memory_turns)\n"
            "print(\"enable_thought_phase:\", o.enable_thought_phase)\n"
            "print(\"stop_phrase:\", o.stop_phrase)"
        ),
        # --- Step 6 ---
        new_markdown_cell(
            "## Step 6 — Structured model replies (`parse_agent_json_response`)\n"
            "The engine asks models for **one JSON object** per turn with keys `thought`, `say`, "
            "`directed_at` (another agent id or null). Fences ```json ... ``` are tolerated."
        ),
        new_code_cell(
            "from agent_rpg.parse import parse_agent_json_response\n"
            "\n"
            "sample = '''```json\n"
            '{"thought":"They are bluffing.", "say":"Show the ledger.", "directed_at": "bob"}\n'
            "```'''\n"
            "p = parse_agent_json_response(sample)\n"
            "print(p.model_dump())"
        ),
        # --- Step 7 ---
        new_markdown_cell(
            "## Step 7 — LLM backends\n"
            "- **`FakeLLMBackend`** — deterministic tests / no network  \n"
            "- **`HuggingFaceInferenceBackend`** — `HF_TOKEN`, `InferenceClient.chat_completion`  \n"
            "- **`TransformersLocalBackend`** — optional extra `[local]`; per-agent `backend: transformers_local`  \n"
            "\n"
            "The engine picks `local_backend` when an agent declares `transformers_local`."
        ),
        new_code_cell(
            "from agent_rpg.backends.fake import FakeLLMBackend\n"
            "from agent_rpg.backends.hf_inference import HuggingFaceInferenceBackend\n"
            "\n"
            "print(\"FakeLLMBackend:\", FakeLLMBackend)\n"
            "print(\"HuggingFaceInferenceBackend:\", HuggingFaceInferenceBackend)"
        ),
        # --- Step 8 ---
        new_markdown_cell(
            "## Step 8 — Run a tiny traced simulation (`SimulationEngine`)\n"
            "One round, mock replies, JSONL under `runs/`."
        ),
        new_code_cell(
            "import json\n"
            "from agent_rpg import SimulationEngine\n"
            "from agent_rpg.backends.fake import FakeLLMBackend\n"
            "from agent_rpg.loader import load_scenario\n"
            "\n"
            "tiny = load_scenario(ROOT / \"examples\" / \"scenarios\" / \"minimal.yaml\")\n"
            "tiny.orchestration.max_rounds = 1\n"
            "tiny.world.max_rounds = 1\n"
            "tiny.orchestration.enable_thought_phase = False\n"
            "\n"
            "def one_shot(i, msgs):\n"
            "    return json.dumps(\n"
            "        {\"thought\": \"\", \"say\": \"(demo line)\", \"directed_at\": None}\n"
            "    )\n"
            "\n"
            "trace_dir = SimulationEngine(tiny).run(\n"
            "    FakeLLMBackend(factory=one_shot),\n"
            "    output_dir=ROOT / \"runs\",\n"
            "    run_id=\"exemplar_trace\",\n"
            ")\n"
            "print(\"JSONL:\", trace_dir / \"events.jsonl\")"
        ),
        # --- Step 9 ---
        new_markdown_cell(
            "## Step 9 — Inspect JSONL events\n"
            "Canonical **`event_type`** values include `system`, `world_event`, `thought`, `message`, "
            "`error`, `router`, `metric`."
        ),
        new_code_cell(
            "from collections import Counter\n"
            "from pathlib import Path\n"
            "\n"
            "lines = (trace_dir / \"events.jsonl\").read_text(encoding=\"utf-8\").strip().splitlines()\n"
            "import json as _json\n"
            "\n"
            "types = Counter(_json.loads(line)[\"event_type\"] for line in lines)\n"
            "print(dict(types))"
        ),
        # --- Step 10 ---
        new_markdown_cell(
            "## Step 10 — Reports (`ReportBuilder`)\n"
            "Post-hoc aggregation: timeline, per-agent quotes, **Gini** on turn counts, directed edges, "
            "optional **`relationships_config`** when you pass the original `ScenarioConfig`."
        ),
        new_code_cell(
            "from agent_rpg import ReportBuilder\n"
            "\n"
            "rb = ReportBuilder.from_jsonl(trace_dir / \"events.jsonl\")\n"
            "summary = rb.to_dict(scenario=tiny)[\"summary\"]\n"
            "print(summary)"
        ),
        # --- Step 11 CONFIG ---
        new_markdown_cell(
            "## Step 11 — **Your** simulation knobs (`SIMULATION_CONFIG`)\n"
            "Edit this dict, then run the remaining cells.\n"
            "\n"
            "| Key | Meaning |\n"
            "|-----|--------|\n"
            "| `seed` | `int` for reproducible randomness; `None` = different each run |\n"
            "| `num_agents` | `2–8` or `None` (random 2–5) |\n"
            "| `max_rounds` | cap rounds or `None` (random 2–4) |\n"
            "| `turn_order` | `'round_robin'` / `'random'` / `'reactive'` / `None` (random) |\n"
            "| `model_id` | Hub model for every agent in this procedural run |\n"
            "| `enable_thought_phase` | `True` / `False` / `None` (keep generator default) |\n"
            "| `cap_max_new_tokens` | upper bound per agent after generation |\n"
            "| `use_hf_if_token` | if `HF_TOKEN` set, use HF backend; else mock |\n"
            "| `run_id_prefix` | prefix for `runs/<id>/` |\n"
            "| `mock_stop_probability` | `0..1`: mock `say` may inject `stop_phrase` to end early |"
        ),
        new_code_cell(
            "import os\n"
            "\n"
            "SIMULATION_CONFIG = {\n"
            "    \"seed\": 2026,\n"
            "    \"num_agents\": 4,\n"
            "    \"max_rounds\": 3,\n"
            "    \"turn_order\": \"round_robin\",\n"
            "    \"model_id\": \"HuggingFaceH4/zephyr-7b-beta\",\n"
            "    \"enable_thought_phase\": False,\n"
            "    \"cap_max_new_tokens\": 200,\n"
            "    \"use_hf_if_token\": False,\n"
            "    \"run_id_prefix\": \"exemplar_rand\",\n"
            "    \"mock_stop_probability\": 0.2,\n"
            "}\n"
            "\n"
            "SIMULATION_CONFIG"
        ),
        # --- Step 12 build ---
        new_markdown_cell(
            "## Step 12 — Build procedural scenario (`build_random_scenario`)\n"
            "`build_random_scenario` samples setting, events, roster, relationships, orchestration "
            "details, and hyperparameters. Values from **`SIMULATION_CONFIG`** override the RNG where set."
        ),
        new_code_cell(
            "from agent_rpg.random_scenario import build_random_scenario\n"
            "\n"
            "try:\n"
            "    from dotenv import load_dotenv\n"
            "\n"
            "    load_dotenv(ROOT / \".env\")\n"
            "except ImportError:\n"
            "    pass\n"
            "\n"
            "rng_seed = SIMULATION_CONFIG.get(\"seed\")\n"
            "scenario = build_random_scenario(\n"
            "    seed=rng_seed,\n"
            "    num_agents=SIMULATION_CONFIG.get(\"num_agents\"),\n"
            "    max_rounds=SIMULATION_CONFIG.get(\"max_rounds\"),\n"
            "    model_id=SIMULATION_CONFIG[\"model_id\"],\n"
            "    turn_order=SIMULATION_CONFIG.get(\"turn_order\"),\n"
            ")\n"
            "if SIMULATION_CONFIG.get(\"enable_thought_phase\") is not None:\n"
            "    scenario.orchestration.enable_thought_phase = SIMULATION_CONFIG[\"enable_thought_phase\"]\n"
            "cap = SIMULATION_CONFIG.get(\"cap_max_new_tokens\")\n"
            "if cap is not None:\n"
            "    for ag in scenario.agents:\n"
            "        ag.max_new_tokens = min(ag.max_new_tokens, int(cap))\n"
            "\n"
            "print(\"scenario_id:\", scenario.world.scenario_id)\n"
            "print(\"title:\", scenario.world.title)\n"
            "print(\"agents:\", [a.display_name for a in scenario.agents])\n"
            "print(\"turn_order:\", scenario.orchestration.turn_order)\n"
            "print(\"rounds:\", scenario.orchestration.max_rounds or scenario.world.max_rounds)"
        ),
        # --- Step 13 run ---
        new_markdown_cell(
            "## Step 13 — Execute full run + report + plots\n"
            "If `use_hf_if_token` and `HF_TOKEN` are set, this calls the Hub; otherwise **`FakeLLMBackend`** "
            "returns varied JSON. Reactive turn order consumes an extra mock line for the router."
        ),
        new_code_cell(
            "import json\n"
            "import os\n"
            "import random\n"
            "\n"
            "from agent_rpg import SimulationEngine\n"
            "from agent_rpg.backends.fake import FakeLLMBackend\n"
            "from agent_rpg.backends.hf_inference import HuggingFaceInferenceBackend\n"
            "\n"
            "use_hf = bool(SIMULATION_CONFIG.get(\"use_hf_if_token\") and os.environ.get(\"HF_TOKEN\"))\n"
            "first_id = scenario.agents[0].agent_id\n"
            "stop = scenario.orchestration.stop_phrase or \"\"\n"
            "p_stop = float(SIMULATION_CONFIG.get(\"mock_stop_probability\") or 0)\n"
            "run_seed = rng_seed if rng_seed is not None else random.randint(1, 10_000_000)\n"
            "\n"
            "if use_hf:\n"
            "    backend = HuggingFaceInferenceBackend()\n"
            "    print(\"Using HuggingFaceInferenceBackend\")\n"
            "else:\n"
            "\n"
            "    def exemplar_factory(i, msgs):\n"
            "        sysm = (msgs[0].get(\"content\") or \"\") if msgs else \"\"\n"
            "        if \"scene director\" in sysm.lower():\n"
            "            return json.dumps({\"next_agent_id\": first_id})\n"
            "        opts = [\n"
            "            {\"thought\": \"weigh odds\", \"say\": \"State your terms clearly.\", \"directed_at\": None},\n"
            "            {\"thought\": \"\", \"say\": \"I will not be cornered.\", \"directed_at\": None},\n"
            "            {\"thought\": \"bluff\", \"say\": \"The crowd is watching; choose wisely.\", \"directed_at\": None},\n"
            "        ]\n"
            "        r = random.Random(run_seed + i)\n"
            "        row = dict(opts[r.randint(0, len(opts) - 1)])\n"
            "        if stop and p_stop > 0 and r.random() < p_stop:\n"
            "            row = {\"thought\": \"end\", \"say\": stop, \"directed_at\": None}\n"
            "        return json.dumps(row)\n"
            "\n"
            "    backend = FakeLLMBackend(factory=exemplar_factory)\n"
            "    print(\"Using FakeLLMBackend\")\n"
            "\n"
            "run_id = f\"{SIMULATION_CONFIG['run_id_prefix']}_{run_seed}\"\n"
            "out = SimulationEngine(scenario).run(\n"
            "    backend,\n"
            "    output_dir=ROOT / \"runs\",\n"
            "    run_id=run_id,\n"
            "    seed=run_seed if isinstance(run_seed, int) else None,\n"
            ")\n"
            "print(\"Wrote:\", out)"
        ),
        new_code_cell(
            "rb2 = ReportBuilder.from_jsonl(out / \"events.jsonl\")\n"
            "report_md = out / \"report.md\"\n"
            "rb2.write_markdown(report_md, scenario=scenario)\n"
            "full = rb2.to_dict(scenario=scenario)\n"
            "print(\"Report:\", report_md)\n"
            "print(\"Messages:\", full[\"summary\"][\"message_count\"])\n"
            "print(\"Gini turns:\", round(full[\"social_dynamics\"][\"gini_turns\"], 4))"
        ),
        new_code_cell(
            "try:\n"
            "    import matplotlib.pyplot as plt\n"
            "\n"
            "    soc = full[\"social_dynamics\"]\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))\n"
            "    tc = soc[\"turn_counts\"]\n"
            "    axes[0].bar(list(tc.keys()), list(tc.values()), color=\"steelblue\")\n"
            "    axes[0].set_title(\"Messages per agent\")\n"
            "    axes[0].tick_params(axis=\"x\", rotation=35)\n"
            "    de = {k: v for k, v in soc.get(\"directed_edges\", {}).items() if v}\n"
            "    if de:\n"
            "        axes[1].bar(list(de.keys()), list(de.values()), color=\"darkseagreen\")\n"
            "        axes[1].set_title(\"Directed edges\")\n"
            "        axes[1].tick_params(axis=\"x\", rotation=60)\n"
            "    else:\n"
            "        axes[1].axis(\"off\")\n"
            "        axes[1].text(0.5, 0.5, \"No directed_at edges\", ha=\"center\")\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "except Exception as exc:\n"
            "    print(\"Plot skipped:\", exc)"
        ),
        new_markdown_cell(
            "### Next steps\n"
            "- Tweak **`SIMULATION_CONFIG`** and re-run from **Step 12** onward.  \n"
            "- Point **`model_id`** at any Hub chat model your Inference plan supports.  \n"
            "- Copy a finished **`events.jsonl`** into analytics tools, or query the optional SQLite mirror from the CLI.  \n"
            "\n"
            "Regenerate notebook files from the repo: `python scripts/rebuild_notebooks.py`."
        ),
    ]
