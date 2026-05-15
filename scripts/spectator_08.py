"""Notebook ``08_live_hf_spectator.ipynb`` — live HF multi-model run with real-time event feed."""

from __future__ import annotations

from nbformat.v4 import new_code_cell, new_markdown_cell


def build_cells(boot: str) -> list:
    return [
        new_markdown_cell(
            "# 08 Live HF spectator\n"
            "\n"
            "Run a **multi-agent simulation** with **real Hugging Face Inference** models. "
            "Each agent can use a **different Hub instruct model** (rotated or shuffled from your pool). "
            "Events appear **as they happen** via `SimulationEngine.run(..., on_event=...)`.\n"
            "\n"
            "**Requirements:** `HF_TOKEN` (`.env` at repo root, **`hf_token.txt`** first line, or your shell env), "
            "`pip install -e '.[notebooks]'`, and models your account can call (some gated repos need license acceptance on the Hub).\n"
            "\n"
            "> Default base model in the library is **Llama 3.1 8B Instruct**; this notebook uses the "
            "**curated small instruct pool** from `agent_rpg.model_catalog`.\n"
            "\n"
            "> **Import issues after `git pull`?** Use **Kernel → Restart**, then run cells from the top "
            "so Jupyter drops a stale `agent_rpg` from memory.\n"
            "\n"
            "> **HTTP 402 from the Hub?** That is billing / included Inference credits exhausted — see "
            "[HF billing settings](https://huggingface.co/settings/billing), or use mock/local backends for free development runs."
        ),
        new_code_cell(boot),
        new_markdown_cell("## 1 — Imports and environment"),
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
            "if not os.environ.get(\"HF_TOKEN\"):\n"
            '    print("WARNING: HF_TOKEN not set — Inference calls will fail.")\n'
            '    print("Fix: export HF_TOKEN=hf_...  OR  create .env with HF_TOKEN=...  OR  put the token on the first line of hf_token.txt (gitignored).")\n'
            "else:\n"
            '    print("HF_TOKEN is set (value hidden).")'
        ),
        new_markdown_cell(
            "## 2 — Choose models (ipywidgets)\n"
            "- **Pool**: pick one or more Hub ids from the curated list (hold Cmd/Ctrl for multi-select).  \n"
            "- **Strategy**: how to assign models across agents (`rotate` / `shuffle` / `random`).  \n"
            "- **Stream tokens**: print model output token-by-token as it arrives (noisy but vivid)."
        ),
        new_code_cell(
            "import ipywidgets as W\n"
            "from IPython.display import display\n"
            "\n"
            "from agent_rpg.model_catalog import SMALL_INSTRUCT_MODELS\n"
            "\n"
            "labels = [m[\"label\"] for m in SMALL_INSTRUCT_MODELS]\n"
            "ids = [m[\"id\"] for m in SMALL_INSTRUCT_MODELS]\n"
            "label_to_id = dict(zip(labels, ids))\n"
            "\n"
            "pool_sel = W.SelectMultiple(\n"
            "    options=[(lbl, lbl) for lbl in labels],\n"
            "    value=(labels[0],),\n"
            "    rows=min(8, len(labels)),\n"
            "    description=\"Model pool\",\n"
            ")\n"
            "strat = W.Dropdown(\n"
            "    options=[(\"rotate\", \"rotate\"), (\"shuffle\", \"shuffle\"), (\"random\", \"random\")],\n"
            "    value=\"rotate\",\n"
            "    description=\"Strategy\",\n"
            ")\n"
            "rounds = W.IntSlider(value=2, min=1, max=6, description=\"Rounds\")\n"
            "n_agents = W.IntSlider(value=3, min=2, max=6, description=\"Agents\")\n"
            "seed = W.IntText(value=42, description=\"Seed\")\n"
            "stream_tokens = W.Checkbox(value=False, description=\"Stream tokens\")\n"
            "thoughts = W.Checkbox(value=False, description=\"Thought phase\")\n"
            "\n"
            "ui = W.VBox([pool_sel, strat, rounds, n_agents, seed, stream_tokens, thoughts])\n"
            "display(ui)"
        ),
        new_markdown_cell("## 3 — Build scenario and assign per-agent models"),
        new_code_cell(
            "from agent_rpg import SimulationEngine, build_random_scenario\n"
            "from agent_rpg.backends.hf_inference import HuggingFaceInferenceBackend\n"
            "from agent_rpg.multi_model import assign_models_to_agents, set_router_model_if_reactive\n"
            "\n"
            "chosen_labels = tuple(pool_sel.value) if pool_sel.value else (labels[0],)\n"
            "model_pool = [label_to_id[l] for l in chosen_labels]\n"
            "\n"
            "scenario = build_random_scenario(\n"
            "    seed=int(seed.value),\n"
            "    num_agents=int(n_agents.value),\n"
            "    max_rounds=int(rounds.value),\n"
            "    model_id=model_pool[0],\n"
            ")\n"
            "assign_models_to_agents(scenario, model_pool, strategy=strat.value, rng=__import__(\"random\").Random(int(seed.value)))\n"
            "set_router_model_if_reactive(scenario, model_pool[0])\n"
            "scenario.orchestration.enable_thought_phase = bool(thoughts.value)\n"
            "for ag in scenario.agents:\n"
            "    ag.max_new_tokens = min(ag.max_new_tokens, 320)\n"
            "    ag.temperature = min(ag.temperature, 0.85)\n"
            "\n"
            "print(\"World:\", scenario.world.title)\n"
            "for a in scenario.agents:\n"
            '    print(f"  {a.display_name}: {a.model_id}")'
        ),
        new_markdown_cell(
            "## 4 — Spectate the run (live events)\n"
            "`on_event` fires after each log line is written — **messages** and **thoughts** are shown as "
            "Markdown. Optional **token streaming** prints raw deltas under the same cell (stderr-style)."
        ),
        new_code_cell(
            "from IPython.display import Markdown, display\n"
            "\n"
            "\n"
            "def on_event(ev):\n"
            "    if ev.event_type == \"world_event\":\n"
            "        display(Markdown(f\"**World** (round {ev.round}): {ev.payload.get('description', '')}\"))\n"
            "    elif ev.event_type == \"thought\":\n"
            "        aid = ev.agent_id or \"?\"\n"
            "        display(Markdown(f\"_Thought — **{aid}**_: {ev.payload.get('text', '')}\"))\n"
            "    elif ev.event_type == \"message\":\n"
            "        aid = ev.agent_id or \"?\"\n"
            "        say = ev.payload.get(\"text\", \"\")\n"
            "        display(Markdown(f\"**{aid}**: {say}\"))\n"
            "    elif ev.event_type == \"router\":\n"
            "        display(Markdown(f\"_Router_: next speaker hint `{ev.payload.get('chosen')!s}`\"))\n"
            "    elif ev.event_type == \"error\":\n"
            "        display(Markdown(f\"⚠ **error**: `{ev.payload}`\"))\n"
            "    elif ev.event_type == \"system\":\n"
            "        display(Markdown(f\"_System_: `{ev.payload}`\"))\n"
            "\n"
            "backend = HuggingFaceInferenceBackend()\n"
            "llm_extras: dict = {}\n"
            "if stream_tokens.value:\n"
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
