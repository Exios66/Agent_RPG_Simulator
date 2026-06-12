"""Live spectator widget factory (notebook 08 controls)."""

from agent_rpg.live_spectator_ui import build_live_spectator_widgets
from agent_rpg.model_catalog import (
    DEFAULT_HF_INFERENCE_MODEL_ID,
    HF_ROUTER_INSTRUCT_MODEL_IDS,
    hf_router_model_entries,
)


def _models_tab(ui):
    return ui.root.children[0]


def _pool_and_per_boxes(ui):
    models_tab = _models_tab(ui)
    return models_tab.children[4], models_tab.children[5]


def test_build_live_spectator_widgets_has_catalog_labels():
    ui = build_live_spectator_widgets()
    assert len(ui.labels) >= 1
    assert ui.labels[0] in ui.label_to_id
    assert ui.exec_mode.value in ("HF Inference API", "Local Transformers")
    assert len(ui.agent_dds) == 8
    assert ui.root.children and len(ui.root.children) == 4


def test_default_pool_uses_hf_router_model():
    ui = build_live_spectator_widgets()
    selected = tuple(ui.pool_sel.value)
    assert selected
    for label in selected:
        assert ui.label_to_id[label] in HF_ROUTER_INSTRUCT_MODEL_IDS


def test_hf_pool_guard_drops_non_router_models_from_selection():
    ui = build_live_spectator_widgets()
    local_only = next(
        label for label in ui.labels if ui.label_to_id[label] not in HF_ROUTER_INSTRUCT_MODEL_IDS
    )
    router_label = hf_router_model_entries()[0]["label"]

    ui.pool_sel.value = (local_only, router_label)
    kept = tuple(ui.pool_sel.value)

    assert kept == (router_label,)
    assert ui.label_to_id[kept[0]] in HF_ROUTER_INSTRUCT_MODEL_IDS


def test_hf_pool_guard_resets_to_default_when_only_non_router_selected():
    ui = build_live_spectator_widgets()
    local_only = next(
        label for label in ui.labels if ui.label_to_id[label] not in HF_ROUTER_INSTRUCT_MODEL_IDS
    )

    ui.pool_sel.value = (local_only,)
    kept = tuple(ui.pool_sel.value)

    assert kept
    assert ui.label_to_id[kept[0]] == DEFAULT_HF_INFERENCE_MODEL_ID


def test_hf_pool_guard_ignored_in_local_transformers_mode():
    ui = build_live_spectator_widgets()
    local_only = next(
        label for label in ui.labels if ui.label_to_id[label] not in HF_ROUTER_INSTRUCT_MODEL_IDS
    )

    ui.exec_mode.value = "Local Transformers"
    ui.pool_sel.value = (local_only,)

    assert tuple(ui.pool_sel.value) == (local_only,)


def test_exec_mode_hf_disables_device_map_and_enables_streaming():
    ui = build_live_spectator_widgets()
    ui.exec_mode.value = "HF Inference API"

    assert ui.stream_tokens.disabled is False
    assert ui.device_map.disabled is True


def test_exec_mode_local_disables_streaming_and_enables_device_map():
    ui = build_live_spectator_widgets()
    ui.exec_mode.value = "Local Transformers"

    assert ui.stream_tokens.disabled is True
    assert ui.device_map.disabled is False


def test_assign_mode_toggles_pool_vs_per_agent_visibility():
    ui = build_live_spectator_widgets()
    pool_box, per_box = _pool_and_per_boxes(ui)

    ui.assign_mode.value = "Pool + strategy"
    assert pool_box.layout.display is None
    assert per_box.layout.display == "none"

    ui.assign_mode.value = "Pick per agent"
    assert pool_box.layout.display == "none"
    assert per_box.layout.display is None
