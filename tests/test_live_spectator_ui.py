"""Live spectator widget factory (notebook 08 controls)."""

from agent_rpg.live_spectator_ui import build_live_spectator_widgets, _short
from agent_rpg.model_catalog import HF_ROUTER_INSTRUCT_MODEL_IDS


def test_build_live_spectator_widgets_has_catalog_labels():
    ui = build_live_spectator_widgets()
    assert len(ui.labels) >= 1
    assert ui.labels[0] in ui.label_to_id
    assert ui.exec_mode.value in ("HF Inference API", "Local Transformers")
    assert len(ui.agent_dds) == 8
    assert ui.root.children and len(ui.root.children) == 4


def test_default_pool_uses_hf_router_model():
    ui = build_live_spectator_widgets()
    assert ui.pool_sel.value
    assert ui.label_to_id[ui.pool_sel.value[0]] in HF_ROUTER_INSTRUCT_MODEL_IDS


def test_hf_pool_guard_resets_non_router_selection():
    """Selecting local-only models in HF mode must revert to a router id (regression for #7)."""
    ui = build_live_spectator_widgets()
    local_label = next(
        lbl for lbl, mid in ui.label_to_id.items() if mid not in HF_ROUTER_INSTRUCT_MODEL_IDS
    )
    ui.pool_sel.value = (local_label,)
    assert ui.label_to_id[ui.pool_sel.value[0]] in HF_ROUTER_INSTRUCT_MODEL_IDS


def test_exec_mode_toggles_stream_and_device_controls():
    ui = build_live_spectator_widgets()
    assert ui.stream_tokens.disabled is False
    assert ui.device_map.disabled is True

    ui.exec_mode.value = "Local Transformers"
    assert ui.stream_tokens.disabled is True
    assert ui.device_map.disabled is False

    ui.exec_mode.value = "HF Inference API"
    assert ui.stream_tokens.disabled is False
    assert ui.device_map.disabled is True


def test_short_truncates_long_strings():
    assert _short("short") == "short"
    long = "x" * 60
    out = _short(long)
    assert len(out) == 52
    assert out.endswith("…")
