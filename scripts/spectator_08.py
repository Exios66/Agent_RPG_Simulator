"""Notebook ``08_live_hf_spectator.ipynb`` — HF Inference and/or local Transformers, multi-model spectator run."""

from __future__ import annotations

from textwrap import dedent

from nbformat.v4 import new_code_cell, new_markdown_cell


WIDGET_CELL = dedent(
    '''
    import ipywidgets as W
    from IPython.display import display

    from agent_rpg.model_catalog import SMALL_INSTRUCT_MODELS
    from agent_rpg.random_scenario import (
        ARCHETYPE_OPTIONS,
        GOAL_OPTIONS,
        OCCUPATION_OPTIONS,
        list_world_preset_titles,
    )

    labels = [m["label"] for m in SMALL_INSTRUCT_MODELS]
    label_to_id = {m["label"]: m["id"] for m in SMALL_INSTRUCT_MODELS}
    _widget_options = [(lbl, lbl) for lbl in labels]
    # Single default model (catalog first = Qwen 1.5B): avoids rotating in a second checkpoint for local runs.
    _default_pool = (labels[0],)

    MAX_AGENT_SLOTS = 8

    # --- Scenario ---
    seed = W.IntText(value=42, description="Procedural seed")
    scenario_note = W.HTML(
        "<p><b>Scenario</b> — seed drives procedural names, world events, and (when customizing) "
        "relationship regeneration.</p>"
    )

    # --- Environment ---
    _titles = list_world_preset_titles()
    world_preset = W.Dropdown(
        options=[("Random premise", "")] + [(t, t) for t in _titles],
        value="",
        description="World premise",
        layout=W.Layout(width="95%"),
    )
    custom_setting = W.Textarea(
        value="",
        placeholder="Optional: override world.setting after generation (empty = keep preset).",
        description="Setting",
        rows=3,
        layout=W.Layout(width="95%"),
    )
    custom_backstory = W.Textarea(
        value="",
        placeholder="Optional: override world.backstory (empty = keep preset).",
        description="Backstory",
        rows=3,
        layout=W.Layout(width="95%"),
    )

    # --- Characters ---
    n_agents = W.IntSlider(value=3, min=2, max=MAX_AGENT_SLOTS, description="Agents")
    char_random = W.Checkbox(
        value=True,
        description="Random roster",
        tooltip="If off, use per-slot display name (if non-empty), occupation, archetype, and goal.",
    )

    def _dd_str(options: tuple[str, ...], desc: str) -> W.Dropdown:
        opts = [(x, x) for x in options]
        return W.Dropdown(options=opts, value=options[0], description=desc, layout=W.Layout(width="360px"))

    char_names = [
        W.Text(value="", placeholder=f"Agent {i + 1} display name", description=f"Name {i + 1}", layout=W.Layout(width="280px"))
        for i in range(MAX_AGENT_SLOTS)
    ]
    char_occ = [_dd_str(OCCUPATION_OPTIONS, f"Job {i + 1}") for i in range(MAX_AGENT_SLOTS)]
    char_arch = [_dd_str(ARCHETYPE_OPTIONS, f"Arch {i + 1}") for i in range(MAX_AGENT_SLOTS)]
    char_goal = [_dd_str(GOAL_OPTIONS, f"Goal {i + 1}") for i in range(MAX_AGENT_SLOTS)]
    char_rows = list(zip(char_names, char_occ, char_arch, char_goal))

    char_grid = W.VBox(
        [W.HTML("<b>Per-slot cast</b> (first <i>N</i> agents; empty name keeps generated name).")]
        + [
            W.HBox(
                [char_names[i], char_occ[i], char_arch[i], char_goal[i]],
                layout=W.Layout(flex_flow="row wrap", align_items="flex-end"),
            )
            for i in range(MAX_AGENT_SLOTS)
        ]
    )

    # --- Parameters ---
    rounds = W.IntSlider(value=2, min=1, max=8, description="Rounds")
    turn_order = W.Dropdown(
        options=[("round_robin", "round_robin"), ("random", "random"), ("reactive", "reactive")],
        value="round_robin",
        description="Turn order",
    )
    memory_turns = W.BoundedIntText(value=20, min=0, max=500, description="Memory turns")
    thoughts = W.Checkbox(value=False, description="Thought phase (global)")
    stop_mode = W.Dropdown(
        options=[
            ("No early stop", "none"),
            ("Custom phrase", "custom"),
            ("Random legacy stop", "sample"),
        ],
        value="none",
        description="Stop phrase",
    )
    stop_phrase_text = W.Text(value="SCENE_END", description="Phrase text", layout=W.Layout(width="280px"))
    max_tokens_cap = W.BoundedIntText(value=320, min=64, max=512, description="Max new tokens cap")
    temp_cap = W.FloatSlider(value=0.85, min=0.1, max=1.5, step=0.05, readout_format=".2f", description="Temperature cap")

    exec_mode = W.ToggleButtons(
        options=["HF Inference API", "Local Transformers"],
        value="Local Transformers",
        description="Execution",
    )
    assign_mode = W.ToggleButtons(
        options=["Pool + strategy", "Pick per agent"],
        description="Assignment",
    )
    pool_sel = W.SelectMultiple(
        options=_widget_options,
        value=_default_pool,
        rows=min(8, len(labels)),
        description="Model pool",
    )
    strat = W.Dropdown(
        options=[("rotate", "rotate"), ("shuffle", "shuffle"), ("random", "random")],
        value="rotate",
        description="Strategy",
    )
    agent_dds = [
        W.Dropdown(options=_widget_options, value=labels[0], description=f"Agent {i + 1}")
        for i in range(MAX_AGENT_SLOTS)
    ]
    stream_tokens = W.Checkbox(value=False, description="Stream tokens (HF only)")
    device_map = W.Text(value="auto", description="Device (local)")

    pool_box = W.VBox(
        [
            W.HTML("<b>Pool</b> — Cmd/Ctrl+click for several models; strategy spreads them across agents."),
            pool_sel,
            strat,
        ]
    )
    per_box = W.VBox([W.HTML("<b>Per agent</b> — one model per row (unused rows ignored).")] + agent_dds)

    def _apply_assign_visibility(_=None):
        use_pool = assign_mode.value == "Pool + strategy"
        pool_box.layout.display = None if use_pool else "none"
        per_box.layout.display = "none" if use_pool else None

    def _apply_agent_rows(_=None):
        n = int(n_agents.value)
        for i, dd in enumerate(agent_dds):
            dd.layout.display = None if i < n else "none"

    def _apply_exec_mode(_=None):
        loc = exec_mode.value == "Local Transformers"
        stream_tokens.disabled = loc
        device_map.disabled = not loc

    def _apply_stop_mode(_=None):
        stop_phrase_text.disabled = stop_mode.value != "custom"

    def _apply_char_visibility(_=None):
        custom = not char_random.value
        n = int(n_agents.value)
        for i, row in enumerate(char_rows):
            vis = custom and i < n
            disp = None if vis else "none"
            for w in row:
                w.layout.display = disp

    assign_mode.observe(_apply_assign_visibility, names="value")
    n_agents.observe(_apply_agent_rows, names="value")
    n_agents.observe(_apply_char_visibility, names="value")
    exec_mode.observe(_apply_exec_mode, names="value")
    char_random.observe(_apply_char_visibility, names="value")
    stop_mode.observe(_apply_stop_mode, names="value")

    _apply_assign_visibility()
    _apply_agent_rows()
    _apply_exec_mode()
    _apply_stop_mode()
    _apply_char_visibility()

    scenario_tab = W.VBox([scenario_note, seed])
    env_tab = W.VBox(
        [
            W.HTML("<b>Environment</b> — premise from procedural library; optional text overrides."),
            world_preset,
            custom_setting,
            custom_backstory,
        ]
    )
    char_tab = W.VBox(
        [
            W.HTML("<b>Characters</b> — agent count and optional manual cast."),
            n_agents,
            char_random,
            char_grid,
        ]
    )
    param_tab = W.VBox(
        [
            W.HTML("<b>Simulation parameters</b>"),
            rounds,
            turn_order,
            memory_turns,
            thoughts,
            stop_mode,
            stop_phrase_text,
            max_tokens_cap,
            temp_cap,
            W.HTML("<b>Inference</b>"),
            exec_mode,
            assign_mode,
            pool_box,
            per_box,
            stream_tokens,
            device_map,
        ]
    )

    tabs = W.Tab(children=[scenario_tab, env_tab, char_tab, param_tab])
    tabs.set_title(0, "Scenario")
    tabs.set_title(1, "Environment")
    tabs.set_title(2, "Characters")
    tabs.set_title(3, "Parameters")
    display(tabs)
    '''
).strip()


def build_cells(boot: str) -> list:
    return [
        new_markdown_cell(
            "# 08 Live HF spectator\n"
            "\n"
            "Run a **multi-agent simulation** with **per-agent Hub `model_id`** values. Choose **how** models are picked "
            "and **where** inference runs:\n"
            "\n"
            "- **HF Inference API** — needs `HF_TOKEN` (`.env` or `hf_token.txt`); uses the curated **open-access** pool "
            "from `agent_rpg.model_catalog` (no Meta Llama gating).\n"
            "- **Local Transformers** — runs on this machine (`pip install -e '.[local]'`); first run **downloads** each "
            "selected checkpoint from the Hub (open models; token optional but helps with rate limits). **Defaults** in "
            "the widgets tab use **Local Transformers** and the **first catalog model** (Qwen 1.5B Instruct); pre-cache "
            "with `python scripts/download_local_instruct_model.py` if you want weights ahead of time.\n"
            "\n"
            "**Tabs:** **Scenario** (seed), **Environment** (premise + optional overrides), **Characters** (count + optional "
            "cast), **Parameters** (orchestration, stop phrase, caps, model assignment, device).\n"
            "\n"
            "> **Import issues after `git pull`?** Kernel → Restart, then run from the top.\n"
            "\n"
            "> **HTTP 402?** HF Inference billing — see [billing](https://huggingface.co/settings/billing) or switch to "
            "**Local Transformers**."
        ),
        new_code_cell(boot),
        new_markdown_cell("## 1 — Environment"),
        new_code_cell(
            "import os\n"
            "\n"
            "try:\n"
            "    from dotenv import load_dotenv\n"
            "\n"
            "    load_dotenv(ROOT / \".env\")\n"
            "except ImportError:\n"
            "    pass\n"
            "\n"
            "if not os.environ.get(\"HF_TOKEN\"):\n"
            "    for _name in (\"hf_token.txt\", \"HF_TOKEN.txt\"):\n"
            "        _p = ROOT / _name\n"
            "        if _p.is_file():\n"
            "            _line = (_p.read_text(encoding=\"utf-8\").strip().splitlines() or [\"\"])[0].strip()\n"
            "            if _line:\n"
            '                os.environ["HF_TOKEN"] = _line\n'
            "            break\n"
            "\n"
            "if os.environ.get(\"HF_TOKEN\"):\n"
            '    print("HF_TOKEN is set (value hidden).")\n'
            "else:\n"
            '    print("HF_TOKEN not set — required for HF Inference API; optional for local downloads (rate limits).")'
        ),
        new_markdown_cell(
            "## 2 — Controls (ipywidgets)\n"
            "Use the **tabs** above the sliders. Re-run **section 3** after changing configuration so `scenario` picks up "
            "new values."
        ),
        new_code_cell(WIDGET_CELL),
        new_markdown_cell("## 3 — Build scenario and wire models / backends"),
        new_code_cell(
            "from agent_rpg import SimulationEngine, build_random_scenario\n"
            "from agent_rpg.backends.hf_inference import HuggingFaceInferenceBackend\n"
            "from agent_rpg.multi_model import assign_models_to_agents, set_router_model_if_reactive\n"
            "from agent_rpg.random_scenario import refresh_agent_relationships\n"
            "\n"
            "_wt = (world_preset.value or \"\").strip()\n"
            "_build_kw = {\n"
            "    \"seed\": int(seed.value),\n"
            "    \"num_agents\": int(n_agents.value),\n"
            "    \"max_rounds\": int(rounds.value),\n"
            "    \"model_id\": label_to_id[labels[0]],\n"
            "    \"turn_order\": turn_order.value,\n"
            "    \"enable_thought_phase\": bool(thoughts.value),\n"
            "    \"memory_turns\": int(memory_turns.value),\n"
            "}\n"
            "if _wt:\n"
            '    _build_kw["world_title"] = _wt\n'
            "\n"
            "_sm = stop_mode.value\n"
            'if _sm == "sample":\n'
            '    _build_kw["sample_stop_phrase"] = True\n'
            '    _build_kw["stop_phrase"] = ""\n'
            'elif _sm == "custom":\n'
            '    _build_kw["sample_stop_phrase"] = False\n'
            '    _build_kw["stop_phrase"] = stop_phrase_text.value.strip()\n'
            "else:\n"
            '    _build_kw["sample_stop_phrase"] = False\n'
            '    _build_kw["stop_phrase"] = ""\n'
            "\n"
            "scenario = build_random_scenario(**_build_kw)\n"
            "\n"
            "_cs = custom_setting.value.strip()\n"
            "if _cs:\n"
            "    scenario.world.setting = _cs\n"
            "_cb = custom_backstory.value.strip()\n"
            "if _cb:\n"
            "    scenario.world.backstory = _cb\n"
            "\n"
            "if not char_random.value:\n"
            "    for i, ag in enumerate(scenario.agents):\n"
            "        nn = (char_names[i].value or \"\").strip()\n"
            "        if nn:\n"
            "            ag.display_name = nn\n"
            "        ag.occupation = char_occ[i].value\n"
            "        ag.archetype = char_arch[i].value\n"
            "        pv = dict(ag.prompt_variables or {})\n"
            '        pv["goal"] = char_goal[i].value\n'
            "        ag.prompt_variables = pv\n"
            "    refresh_agent_relationships(scenario, seed=int(seed.value))\n"
            "\n"
            'if assign_mode.value == "Pick per agent":\n'
            "    for i, ag in enumerate(scenario.agents):\n"
            "        ag.model_id = label_to_id[agent_dds[i].value]\n"
            "else:\n"
            "    chosen_labels = tuple(pool_sel.value) if pool_sel.value else (labels[0],)\n"
            "    model_pool = [label_to_id[l] for l in chosen_labels]\n"
            "    assign_models_to_agents(\n"
            "        scenario,\n"
            "        model_pool,\n"
            "        strategy=strat.value,\n"
            "        rng=__import__(\"random\").Random(int(seed.value)),\n"
            "    )\n"
            "\n"
            "set_router_model_if_reactive(scenario, scenario.agents[0].model_id)\n"
            "\n"
            'if exec_mode.value == "Local Transformers":\n'
            "    for ag in scenario.agents:\n"
            '        ag.backend = "transformers_local"\n'
            "else:\n"
            "    for ag in scenario.agents:\n"
            '        ag.backend = "auto"\n'
            "\n"
            "_cap_tok = int(max_tokens_cap.value)\n"
            "_cap_temp = float(temp_cap.value)\n"
            "for ag in scenario.agents:\n"
            "    ag.max_new_tokens = min(ag.max_new_tokens, _cap_tok)\n"
            "    ag.temperature = min(ag.temperature, _cap_temp)\n"
            "\n"
            'print("World:", scenario.world.title)\n'
            'print("Turn order:", scenario.orchestration.turn_order, "stop:", repr(scenario.orchestration.stop_phrase))\n'
            'print("Execution:", exec_mode.value)\n'
            "for a in scenario.agents:\n"
            '    print(f"  {a.display_name} ({a.occupation}): {a.model_id}  backend={a.backend}")'
        ),
        new_markdown_cell(
            "## 4 — Spectate the run\n"
            "Local runs ignore **Stream tokens** (no streaming path in `TransformersLocalBackend` yet)."
        ),
        new_code_cell(
            "from IPython.display import Markdown, display\n"
            "\n"
            "_agent_by_id = {a.agent_id: a for a in scenario.agents}\n"
            "\n"
            "\n"
            "def _agent_line(aid: str | None) -> str:\n"
            "    if not aid:\n"
            '        return "?"\n'
            "    ag = _agent_by_id.get(aid)\n"
            "    if ag is None:\n"
            "        return aid\n"
            "    occ = (ag.occupation or \"\").strip()\n"
            "    arch = (ag.archetype or \"\").strip()\n"
            "    if occ and arch:\n"
            '        return f"{ag.display_name} ({occ} — {arch})"\n'
            "    if occ:\n"
            '        return f"{ag.display_name} ({occ})"\n'
            "    if arch:\n"
            '        return f"{ag.display_name} ({arch})"\n'
            "    return ag.display_name\n"
            "\n"
            "\n"
            "def on_event(ev):\n"
            "    if ev.event_type == \"world_event\":\n"
            "        eid = ev.payload.get(\"event_id\", \"\")\n"
            "        desc = ev.payload.get(\"description\", \"\")\n"
            "        sub = f\"`{eid}` — \" if eid else \"\"\n"
            "        display(Markdown(f\"**World** (round {ev.round}): {sub}{desc}\"))\n"
            "    elif ev.event_type == \"thought\":\n"
            "        who = _agent_line(ev.agent_id)\n"
            "        display(Markdown(f\"_Thought — **{who}**_: {ev.payload.get('text', '')}\"))\n"
            "    elif ev.event_type == \"message\":\n"
            "        who = _agent_line(ev.agent_id)\n"
            "        say = ev.payload.get(\"text\", \"\")\n"
            "        display(Markdown(f\"**{who}**: {say}\"))\n"
            "    elif ev.event_type == \"router\":\n"
            "        display(Markdown(f\"_Router_: next speaker hint `{ev.payload.get('chosen')!s}`\"))\n"
            "    elif ev.event_type == \"error\":\n"
            "        display(Markdown(f\"⚠ **error**: `{ev.payload}`\"))\n"
            "    elif ev.event_type == \"system\":\n"
            "        display(Markdown(f\"_System_: `{ev.payload}`\"))\n"
            "\n"
            "local_backend = None\n"
            '_use_local = exec_mode.value == "Local Transformers"\n'
            "if _use_local:\n"
            "    from agent_rpg.backends.transformers_local import TransformersLocalBackend\n"
            "\n"
            "    _dm = device_map.value.strip() or \"auto\"\n"
            "    local_backend = TransformersLocalBackend(device_map=_dm)\n"
            "\n"
            "backend = HuggingFaceInferenceBackend()\n"
            "llm_extras: dict = {}\n"
            "if stream_tokens.value and not _use_local:\n"
            "\n"
            "    def chunk_cb(t: str) -> None:\n"
            '        print(t, end="", flush=True)\n'
            "\n"
            "    llm_extras = {\"stream\": True, \"chunk_callback\": chunk_cb}\n"
            "\n"
            "run_id = f\"live_spec_{int(seed.value)}\"\n"
            "print(\"--- simulation start ---\\n\")\n"
            "out = SimulationEngine(scenario).run(\n"
            "    backend,\n"
            "    output_dir=ROOT / \"runs\",\n"
            "    run_id=run_id,\n"
            "    seed=int(seed.value),\n"
            "    on_event=on_event,\n"
            "    llm_extras=llm_extras,\n"
            "    local_backend=local_backend,\n"
            ")\n"
            'print("\\n--- done ---\\nJSONL:", out / "events.jsonl")'
        ),
        new_markdown_cell("## 5 — Optional: quick report"),
        new_code_cell(
            "from agent_rpg import ReportBuilder\n"
            "\n"
            "rb = ReportBuilder.from_jsonl(out / \"events.jsonl\")\n"
            "d = rb.to_dict(scenario=scenario)\n"
            'print("Messages:", d["summary"]["message_count"], "Errors:", d["summary"]["error_count"])\n'
            'print("Gini:", round(d["social_dynamics"]["gini_turns"], 4))'
        ),
    ]
