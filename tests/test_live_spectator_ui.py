"""Live spectator widget factory (notebook 08 controls)."""

from __future__ import annotations

import pytest

from agent_rpg.live_spectator_ui import build_live_spectator_widgets
from agent_rpg.model_catalog import (
    DEFAULT_INSTRUCT_MODEL_ID,
    HF_ROUTER_INSTRUCT_MODEL_IDS,
    hf_router_model_entries,
)


def _label_for_model_id(ui, model_id: str) -> str:
    for label, mid in ui.label_to_id.items():
        if mid == model_id:
            return label
    pytest.fail(f"no catalog label for {model_id}")


def test_build_live_spectator_widgets_has_catalog_labels():
    ui = build_live_spectator_widgets()
    assert len(ui.labels) >= 1
    assert ui.labels[0] in ui.label_to_id
    assert ui.exec_mode.value in ("HF Inference API", "Local Transformers")
    assert len(ui.agent_dds) == 8
    assert ui.root.children and len(ui.root.children) == 4


def test_hf_pool_guard_replaces_non_router_selection() -> None:
    """HF mode must not leave 1.5B/3B in the pool (router returns model_not_supported)."""
    ui = build_live_spectator_widgets()
    local_label = _label_for_model_id(ui, DEFAULT_INSTRUCT_MODEL_ID)
    ui.exec_mode.value = "HF Inference API"
    ui.pool_sel.value = (local_label,)
    selected_ids = {ui.label_to_id[lbl] for lbl in ui.pool_sel.value}
    assert selected_ids <= HF_ROUTER_INSTRUCT_MODEL_IDS


def test_hf_pool_guard_keeps_router_models_only() -> None:
    ui = build_live_spectator_widgets()
    local_label = _label_for_model_id(ui, DEFAULT_INSTRUCT_MODEL_ID)
    hf_label = hf_router_model_entries()[0]["label"]
    ui.exec_mode.value = "HF Inference API"
    ui.pool_sel.value = (local_label, hf_label)
    assert ui.pool_sel.value == (hf_label,)


def test_hf_pool_guard_noop_in_local_mode() -> None:
    ui = build_live_spectator_widgets()
    local_label = _label_for_model_id(ui, DEFAULT_INSTRUCT_MODEL_ID)
    ui.exec_mode.value = "Local Transformers"
    ui.pool_sel.value = (local_label,)
    assert ui.pool_sel.value == (local_label,)


def test_exec_mode_toggles_stream_and_device_controls() -> None:
    ui = build_live_spectator_widgets()
    ui.exec_mode.value = "Local Transformers"
    assert ui.stream_tokens.disabled is True
    assert ui.device_map.disabled is False
    ui.exec_mode.value = "HF Inference API"
    assert ui.stream_tokens.disabled is False
    assert ui.device_map.disabled is True
