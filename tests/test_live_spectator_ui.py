"""Live spectator widget factory (notebook 08 controls)."""

from agent_rpg.live_spectator_ui import _short, build_live_spectator_widgets
from agent_rpg.model_catalog import HF_ROUTER_INSTRUCT_MODEL_IDS


def test_build_live_spectator_widgets_has_catalog_labels():
    ui = build_live_spectator_widgets()
    assert len(ui.labels) >= 1
    assert ui.labels[0] in ui.label_to_id
    assert ui.exec_mode.value in ("HF Inference API", "Local Transformers")
    assert len(ui.agent_dds) == 8
    assert ui.root.children and len(ui.root.children) == 4


def test_short_truncates_long_strings():
    assert _short("hello", n=10) == "hello"
    assert _short("x" * 60, n=10) == ("x" * 9) + "…"


def test_hf_pool_guard_resets_non_router_models():
    """HF Inference mode must not keep local-only models in the pool (model_not_supported)."""
    ui = build_live_spectator_widgets()
    local_label = next(
        lbl for lbl in ui.labels if ui.label_to_id[lbl] not in HF_ROUTER_INSTRUCT_MODEL_IDS
    )
    ui.pool_sel.value = (local_label,)
    selected_ids = {ui.label_to_id[lbl] for lbl in ui.pool_sel.value}
    assert selected_ids <= HF_ROUTER_INSTRUCT_MODEL_IDS


def test_hf_pool_guard_filters_mixed_selection():
    ui = build_live_spectator_widgets()
    local_label = next(
        lbl for lbl in ui.labels if ui.label_to_id[lbl] not in HF_ROUTER_INSTRUCT_MODEL_IDS
    )
    router_label = next(
        lbl for lbl in ui.labels if ui.label_to_id[lbl] in HF_ROUTER_INSTRUCT_MODEL_IDS
    )
    ui.pool_sel.value = (local_label, router_label)
    assert ui.pool_sel.value == (router_label,)


def test_exec_mode_toggles_stream_and_device_controls():
    ui = build_live_spectator_widgets()
    ui.exec_mode.value = "Local Transformers"
    assert ui.stream_tokens.disabled is True
    assert ui.device_map.disabled is False

    ui.exec_mode.value = "HF Inference API"
    assert ui.stream_tokens.disabled is False
    assert ui.device_map.disabled is True


def test_assign_mode_defaults_to_pool_strategy():
    ui = build_live_spectator_widgets()
    assert ui.assign_mode.value == "Pool + strategy"
    assert ui.strat.value == "rotate"
