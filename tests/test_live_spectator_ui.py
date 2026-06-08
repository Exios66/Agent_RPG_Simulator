"""Live spectator widget factory (notebook 08 controls)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent_rpg.live_spectator_ui import (
    DISPLAY_ID,
    _short,
    build_live_spectator_widgets,
    display_live_spectator,
    ensure_ipywidgets,
)
from agent_rpg.model_catalog import (
    DEFAULT_INSTRUCT_MODEL_ID,
    HF_ROUTER_INSTRUCT_MODEL_IDS,
    hf_router_model_entries,
    labels_by_id,
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


def test_short_truncates_long_strings():
    assert _short("short") == "short"
    assert _short("x" * 52) == "x" * 52
    assert _short("x" * 60) == "x" * 51 + "…"
    assert len(_short("y" * 100)) == 52


def test_default_pool_uses_hf_router_model():
    ui = build_live_spectator_widgets()
    hf_labels = {m["label"] for m in hf_router_model_entries()}
    assert ui.pool_sel.value
    for label in ui.pool_sel.value:
        assert label in hf_labels
        assert ui.label_to_id[label] in HF_ROUTER_INSTRUCT_MODEL_IDS


def test_hf_pool_guard_drops_non_router_selection():
    ui = build_live_spectator_widgets()
    local_label = labels_by_id()[DEFAULT_INSTRUCT_MODEL_ID]
    assert ui.label_to_id[local_label] not in HF_ROUTER_INSTRUCT_MODEL_IDS

    ui.pool_sel.value = (local_label,)
    for label in ui.pool_sel.value:
        assert ui.label_to_id[label] in HF_ROUTER_INSTRUCT_MODEL_IDS


def test_hf_pool_guard_keeps_router_subset_when_mixed():
    ui = build_live_spectator_widgets()
    router_label = hf_router_model_entries()[0]["label"]
    local_label = labels_by_id()[DEFAULT_INSTRUCT_MODEL_ID]

    ui.pool_sel.value = (router_label, local_label)
    assert router_label in ui.pool_sel.value
    assert local_label not in ui.pool_sel.value


def test_exec_mode_toggles_stream_and_device_controls():
    ui = build_live_spectator_widgets()
    assert ui.exec_mode.value == "HF Inference API"
    assert ui.stream_tokens.disabled is False
    assert ui.device_map.disabled is True

    ui.exec_mode.value = "Local Transformers"
    assert ui.stream_tokens.disabled is True
    assert ui.device_map.disabled is False


def test_assign_mode_toggles_pool_vs_per_agent_panels():
    ui = build_live_spectator_widgets()
    pool_box, per_box = _pool_and_per_boxes(ui)
    assert ui.assign_mode.value == "Pool + strategy"
    assert pool_box.layout.display is None
    assert per_box.layout.display == "none"

    ui.assign_mode.value = "Pick per agent"
    assert pool_box.layout.display == "none"
    assert per_box.layout.display is None


def test_n_agents_hides_unused_agent_dropdowns():
    ui = build_live_spectator_widgets()
    ui.assign_mode.value = "Pick per agent"
    ui.n_agents.value = 3
    for i, dd in enumerate(ui.agent_dds):
        if i < 3:
            assert dd.layout.display is None
        else:
            assert dd.layout.display == "none"


def test_display_live_spectator_falls_back_when_update_display_fails():
    ui = build_live_spectator_widgets()
    with (
        patch("agent_rpg.live_spectator_ui.W.Widget.close_all") as close_all,
        patch("agent_rpg.live_spectator_ui.clear_output") as clear_output,
        patch(
            "agent_rpg.live_spectator_ui.update_display",
            side_effect=RuntimeError("no active display"),
        ),
        patch("agent_rpg.live_spectator_ui.display") as display_fn,
    ):
        display_live_spectator(ui)

    close_all.assert_called_once()
    clear_output.assert_called_once_with(wait=True)
    display_fn.assert_called_once_with(ui.root, display_id=DISPLAY_ID)


def test_ensure_ipywidgets_raises_with_install_hint():
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ipywidgets":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(ImportError, match="requirements-notebooks"):
            ensure_ipywidgets()
