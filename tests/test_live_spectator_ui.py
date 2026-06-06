"""Live spectator widget factory (notebook 08 controls)."""

from agent_rpg.live_spectator_ui import build_live_spectator_widgets
from agent_rpg.model_catalog import HF_ROUTER_INSTRUCT_MODEL_IDS


def _non_router_label(ui) -> str:
    return next(
        lbl for lbl in ui.labels if ui.label_to_id[lbl] not in HF_ROUTER_INSTRUCT_MODEL_IDS
    )


def _router_label(ui) -> str:
    return next(lbl for lbl in ui.labels if ui.label_to_id[lbl] in HF_ROUTER_INSTRUCT_MODEL_IDS)


def test_build_live_spectator_widgets_has_catalog_labels():
    ui = build_live_spectator_widgets()
    assert len(ui.labels) >= 1
    assert ui.labels[0] in ui.label_to_id
    assert ui.exec_mode.value in ("HF Inference API", "Local Transformers")
    assert len(ui.agent_dds) == 8
    assert ui.root.children and len(ui.root.children) == 4


def test_hf_pool_defaults_to_router_model():
    ui = build_live_spectator_widgets()
    assert ui.exec_mode.value == "HF Inference API"
    assert ui.pool_sel.value
    for label in ui.pool_sel.value:
        assert ui.label_to_id[label] in HF_ROUTER_INSTRUCT_MODEL_IDS


def test_hf_pool_guard_strips_non_router_models():
    ui = build_live_spectator_widgets()
    local_label = _non_router_label(ui)
    ui.pool_sel.value = (local_label,)
    ui.exec_mode.value = "HF Inference API"
    for label in ui.pool_sel.value:
        assert ui.label_to_id[label] in HF_ROUTER_INSTRUCT_MODEL_IDS


def test_hf_pool_guard_keeps_router_models():
    ui = build_live_spectator_widgets()
    router_label = _router_label(ui)
    ui.pool_sel.value = (router_label,)
    ui.exec_mode.value = "HF Inference API"
    assert ui.pool_sel.value == (router_label,)


def test_local_exec_mode_allows_non_router_pool():
    ui = build_live_spectator_widgets()
    local_label = _non_router_label(ui)
    ui.exec_mode.value = "Local Transformers"
    ui.pool_sel.value = (local_label,)
    assert ui.pool_sel.value == (local_label,)


def test_exec_mode_toggles_stream_and_device_controls():
    ui = build_live_spectator_widgets()
    ui.exec_mode.value = "HF Inference API"
    assert ui.stream_tokens.disabled is False
    assert ui.device_map.disabled is True
    ui.exec_mode.value = "Local Transformers"
    assert ui.stream_tokens.disabled is True
    assert ui.device_map.disabled is False
