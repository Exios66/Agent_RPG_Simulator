"""ipywidgets UI for ``notebooks/08_live_hf_spectator.ipynb``.

Uses a single top-level ``Tab`` widget (better rendering in VS Code / Cursor than a deep
``VBox`` tree) and clears stale comms before each display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ipywidgets as W
from IPython.display import clear_output, display, update_display

from agent_rpg.model_catalog import (
    HF_ROUTER_INSTRUCT_MODEL_IDS,
    SMALL_INSTRUCT_MODELS,
    default_model_id_for_execution,
    hf_router_model_entries,
    label_to_id_map,
)
from agent_rpg.random_scenario import (
    ARCHETYPE_OPTIONS,
    GOAL_OPTIONS,
    OCCUPATION_OPTIONS,
    list_world_preset_titles,
)

DISPLAY_ID = "live_spectator_controls"
MAX_AGENT_SLOTS = 8

# Wider controls help VS Code / Cursor widget hosts avoid zero-width children.
# Each widget needs its own Layout instance so per-row display toggles do not leak.
def _ctrl_layout() -> W.Layout:
    return W.Layout(width="auto", min_width="28em")


def _wide_layout() -> W.Layout:
    return W.Layout(width="auto", min_width="36em", max_width="100%")


@dataclass
class LiveSpectatorWidgets:
    """Handles to every control; downstream cells read ``.value`` on each widget."""

    exec_mode: W.ToggleButtons
    assign_mode: W.ToggleButtons
    pool_sel: W.SelectMultiple
    strat: W.Dropdown
    agent_dds: list[W.Dropdown]
    n_agents: W.IntSlider
    rounds: W.IntSlider
    seed: W.IntText
    stream_tokens: W.Checkbox
    thoughts: W.Checkbox
    device_map: W.Text
    world_premise_dd: W.Dropdown
    env_setting_ta: W.Textarea
    env_backstory_ta: W.Textarea
    env_notes_ta: W.Textarea
    char_goal_dd: W.Dropdown
    char_arch_dd: W.Dropdown
    char_occ_dd: W.Dropdown
    turn_order_dd: W.Dropdown
    memory_turns_sld: W.IntSlider
    stop_phrase_txt: W.Text
    labels: list[str]
    label_to_id: dict[str, str]
    root: W.Tab


def _short(s: str, n: int = 52) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def ensure_ipywidgets() -> None:
    """Print a short diagnostic; raise if ipywidgets is missing."""
    try:
        import ipywidgets  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "ipywidgets is required for notebook 08 controls. "
            "Install with: pip install -r requirements-notebooks.txt"
        ) from e
    import ipywidgets as iw

    ver = getattr(iw, "__version__", "?")
    print(f"ipywidgets {ver} — if controls show only text, re-run this cell after Kernel → Restart.")


def build_live_spectator_widgets() -> LiveSpectatorWidgets:
    """Construct widgets without displaying (for tests)."""
    labels = [m["label"] for m in SMALL_INSTRUCT_MODELS]
    label_to_id = label_to_id_map()
    hf_labels = [m["label"] for m in hf_router_model_entries()]
    hf_default_label = hf_labels[0] if hf_labels else (labels[0] if labels else "")
    default_pool = (hf_default_label,) if hf_default_label else tuple()

    exec_mode = W.ToggleButtons(
        options=["HF Inference API", "Local Transformers"],
        value="HF Inference API",
        description="Execution",
        layout=_ctrl_layout(),
    )
    assign_mode = W.ToggleButtons(
        options=["Pool + strategy", "Pick per agent"],
        value="Pool + strategy",
        description="Assignment",
        layout=_ctrl_layout(),
    )
    pool_sel = W.SelectMultiple(
        options=labels,
        value=default_pool,
        rows=min(8, max(4, len(labels))),
        description="Model pool",
        layout=_ctrl_layout(),
    )
    strat = W.Dropdown(
        options=[("rotate", "rotate"), ("shuffle", "shuffle"), ("random", "random")],
        value="rotate",
        description="Strategy",
        layout=_ctrl_layout(),
    )

    agent_dds = [
        W.Dropdown(
            options=labels,
            value=(hf_labels[i % len(hf_labels)] if hf_labels else labels[i % len(labels)])
            if labels
            else None,
            description=f"Agent {i + 1}",
            layout=_ctrl_layout(),
        )
        for i in range(MAX_AGENT_SLOTS)
    ]

    n_agents = W.IntSlider(value=3, min=2, max=8, description="Agents", layout=_ctrl_layout())
    rounds = W.IntSlider(value=2, min=1, max=6, description="Rounds", layout=_ctrl_layout())
    seed = W.IntText(value=42, description="Seed", layout=_ctrl_layout())
    stream_tokens = W.Checkbox(value=False, description="Stream tokens (HF only)")
    thoughts = W.Checkbox(value=False, description="Thought phase")
    device_map = W.Text(value="auto", description="Device (local)", layout=_ctrl_layout())

    preset_titles = list_world_preset_titles()
    world_dd_opts = [("(Random premise)", "")] + [(t, t) for t in preset_titles]
    world_premise_dd = W.Dropdown(
        options=world_dd_opts,
        value="",
        description="World premise",
        layout=_wide_layout(),
    )
    env_setting_ta = W.Textarea(
        value="",
        placeholder="Optional: replace sensory scene text (leave blank for preset)",
        rows=2,
        description="Setting",
        layout=_wide_layout(),
    )
    env_backstory_ta = W.Textarea(
        value="",
        placeholder="Optional: replace central conflict (leave blank for preset)",
        rows=2,
        description="Backstory",
        layout=_wide_layout(),
    )
    env_notes_ta = W.Textarea(
        value="",
        placeholder="Optional: replace world rules / notes (leave blank for preset)",
        rows=2,
        description="World notes",
        layout=_wide_layout(),
    )

    char0 = ("(random)", "")
    char_goal_dd = W.Dropdown(
        options=[char0] + [(_short(g), g) for g in GOAL_OPTIONS],
        value="",
        description="Shared goal",
        layout=_wide_layout(),
    )
    char_arch_dd = W.Dropdown(
        options=[char0] + [(a, a) for a in ARCHETYPE_OPTIONS],
        value="",
        description="Archetype",
        layout=_ctrl_layout(),
    )
    char_occ_dd = W.Dropdown(
        options=[char0] + [(o, o) for o in OCCUPATION_OPTIONS],
        value="",
        description="Occupation",
        layout=_ctrl_layout(),
    )

    turn_order_dd = W.Dropdown(
        options=[
            ("round robin", "round_robin"),
            ("random", "random"),
            ("reactive (router)", "reactive"),
        ],
        value="round_robin",
        description="Turn order",
        layout=_ctrl_layout(),
    )
    memory_turns_sld = W.IntSlider(
        value=16, min=4, max=32, step=2, description="Memory turns", layout=_ctrl_layout()
    )
    stop_phrase_txt = W.Text(value="", description="Stop phrase", layout=_ctrl_layout())

    pool_box = W.VBox(
        [
            W.HTML("<b>Pool</b> — Cmd/Ctrl+click to pick several models; strategy spreads them across agents."),
            pool_sel,
            strat,
        ]
    )
    per_box = W.VBox(
        [W.HTML("<b>Per agent</b> — set a model for each row (unused rows are ignored).")] + agent_dds
    )

    def _apply_assign_visibility(_: Any = None) -> None:
        use_pool = assign_mode.value == "Pool + strategy"
        pool_box.layout.display = None if use_pool else "none"
        per_box.layout.display = "none" if use_pool else None

    def _apply_agent_rows(_: Any = None) -> None:
        n = int(n_agents.value)
        for i, dd in enumerate(agent_dds):
            dd.layout.display = None if i < n else "none"

    hf_pool_hint = W.HTML(
        "<i>HF Inference:</i> use models marked <b>HF router</b> in the pool (e.g. Qwen 7B). "
        "1.5B / 3B often return <code>model_not_supported</code>."
    )

    def _apply_exec_mode(_: Any = None) -> None:
        loc = exec_mode.value == "Local Transformers"
        stream_tokens.disabled = loc
        device_map.disabled = not loc
        hf_pool_hint.layout.display = None if not loc else "none"

    def _apply_hf_pool_guard(_: Any = None) -> None:
        if exec_mode.value != "HF Inference API" or not hf_labels:
            return
        cur = tuple(pool_sel.value) if pool_sel.value else ()
        kept = tuple(x for x in cur if label_to_id.get(x) in HF_ROUTER_INSTRUCT_MODEL_IDS)
        if not kept:
            pool_sel.value = (hf_default_label,)
        elif kept != cur:
            pool_sel.value = kept

    assign_mode.observe(_apply_assign_visibility, names="value")
    n_agents.observe(_apply_agent_rows, names="value")
    exec_mode.observe(_apply_exec_mode, names="value")
    exec_mode.observe(_apply_hf_pool_guard, names="value")
    pool_sel.observe(_apply_hf_pool_guard, names="value")
    _apply_assign_visibility()
    _apply_agent_rows()
    _apply_exec_mode()
    _apply_hf_pool_guard()

    models_tab = W.VBox(
        [
            W.HTML("<b>Models & inference</b>"),
            hf_pool_hint,
            exec_mode,
            assign_mode,
            pool_box,
            per_box,
            stream_tokens,
            device_map,
        ]
    )
    world_tab = W.VBox(
        [
            W.HTML("<b>World</b> — premise picks background-event pool; text areas override only if non-empty."),
            world_premise_dd,
            W.HTML("<i>Environment overrides</i>"),
            env_setting_ta,
            env_backstory_ta,
            env_notes_ta,
        ]
    )
    cast_tab = W.VBox(
        [
            W.HTML(
                "<b>Characters</b> — same goal / archetype / occupation for <i>all</i> agents "
                "(random keeps variety)."
            ),
            char_goal_dd,
            char_arch_dd,
            char_occ_dd,
        ]
    )
    run_tab = W.VBox(
        [
            W.HTML("<b>Run</b> — size, orchestration, and early-exit phrase."),
            n_agents,
            rounds,
            seed,
            turn_order_dd,
            memory_turns_sld,
            stop_phrase_txt,
            thoughts,
        ]
    )

    tabs = W.Tab(children=[models_tab, world_tab, cast_tab, run_tab])
    tabs.set_title(0, "Models")
    tabs.set_title(1, "World")
    tabs.set_title(2, "Characters")
    tabs.set_title(3, "Run")

    return LiveSpectatorWidgets(
        exec_mode=exec_mode,
        assign_mode=assign_mode,
        pool_sel=pool_sel,
        strat=strat,
        agent_dds=agent_dds,
        n_agents=n_agents,
        rounds=rounds,
        seed=seed,
        stream_tokens=stream_tokens,
        thoughts=thoughts,
        device_map=device_map,
        world_premise_dd=world_premise_dd,
        env_setting_ta=env_setting_ta,
        env_backstory_ta=env_backstory_ta,
        env_notes_ta=env_notes_ta,
        char_goal_dd=char_goal_dd,
        char_arch_dd=char_arch_dd,
        char_occ_dd=char_occ_dd,
        turn_order_dd=turn_order_dd,
        memory_turns_sld=memory_turns_sld,
        stop_phrase_txt=stop_phrase_txt,
        labels=labels,
        label_to_id=label_to_id,
        root=tabs,
    )


def display_live_spectator(ui: LiveSpectatorWidgets) -> None:
    """Show ``ui.root``; safe to call again after Kernel restart."""
    W.Widget.close_all()
    clear_output(wait=True)
    try:
        update_display(ui.root, display_id=DISPLAY_ID)
    except Exception:
        display(ui.root, display_id=DISPLAY_ID)


def build_and_display_live_spectator() -> LiveSpectatorWidgets:
    """Build controls, display them, and return handles for later cells."""
    ensure_ipywidgets()
    ui = build_live_spectator_widgets()
    display_live_spectator(ui)
    return ui
