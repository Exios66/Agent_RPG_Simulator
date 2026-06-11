"""Live spectator widget factory (notebook 08 controls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_rpg.live_spectator_ui import (
    DISPLAY_ID,
    _short,
    build_live_spectator_widgets,
    display_live_spectator,
    ensure_ipywidgets,
)
from agent_rpg.model_catalog import HF_ROUTER_INSTRUCT_MODEL_IDS


def _models_tab(ui):
    return ui.root.children[0]


def _pool_box(ui):
    return _models_tab(ui).children[4]


def _per_box(ui):
    return _models_tab(ui).children[5]


def _hf_pool_hint(ui):
    return _models_tab(ui).children[1]


def test_build_live_spectator_widgets_has_catalog_labels():
    ui = build_live_spectator_widgets()
    assert len(ui.labels) >= 1
    assert ui.labels[0] in ui.label_to_id
    assert ui.exec_mode.value in ("HF Inference API", "Local Transformers")
    assert len(ui.agent_dds) == 8
    assert ui.root.children and len(ui.root.children) == 4


def test_short_truncates_long_strings() -> None:
    assert _short("hello", 10) == "hello"
    assert _short("x" * 60, 52) == "x" * 51 + "…"


def test_default_pool_uses_hf_router_model() -> None:
    ui = build_live_spectator_widgets()
    assert ui.pool_sel.value
    for label in ui.pool_sel.value:
        assert ui.label_to_id[label] in HF_ROUTER_INSTRUCT_MODEL_IDS


def test_hf_pool_guard_filters_non_router_models() -> None:
    ui = build_live_spectator_widgets()
    local_only = [
        lbl for lbl in ui.labels if ui.label_to_id[lbl] not in HF_ROUTER_INSTRUCT_MODEL_IDS
    ]
    assert local_only, "catalog should include at least one non-HF-router model"
    ui.exec_mode.value = "HF Inference API"
    ui.pool_sel.value = tuple(local_only[:2])
    kept = tuple(ui.pool_sel.value)
    assert all(ui.label_to_id[lbl] in HF_ROUTER_INSTRUCT_MODEL_IDS for lbl in kept)


def test_hf_pool_guard_noop_in_local_mode() -> None:
    ui = build_live_spectator_widgets()
    local_only = [
        lbl for lbl in ui.labels if ui.label_to_id[lbl] not in HF_ROUTER_INSTRUCT_MODEL_IDS
    ]
    ui.exec_mode.value = "Local Transformers"
    ui.pool_sel.value = (local_only[0],)
    assert ui.pool_sel.value == (local_only[0],)


def test_exec_mode_local_disables_stream_tokens() -> None:
    ui = build_live_spectator_widgets()
    ui.exec_mode.value = "Local Transformers"
    assert ui.stream_tokens.disabled is True
    assert ui.device_map.disabled is False
    assert _hf_pool_hint(ui).layout.display == "none"


def test_exec_mode_hf_enables_stream_tokens() -> None:
    ui = build_live_spectator_widgets()
    ui.exec_mode.value = "HF Inference API"
    assert ui.stream_tokens.disabled is False
    assert ui.device_map.disabled is True
    assert _hf_pool_hint(ui).layout.display is None


def test_assign_mode_pool_shows_pool_box() -> None:
    ui = build_live_spectator_widgets()
    ui.assign_mode.value = "Pool + strategy"
    assert _pool_box(ui).layout.display is None
    assert _per_box(ui).layout.display == "none"


def test_assign_mode_per_agent_shows_per_box() -> None:
    ui = build_live_spectator_widgets()
    ui.assign_mode.value = "Pick per agent"
    assert _pool_box(ui).layout.display == "none"
    assert _per_box(ui).layout.display is None


def test_display_live_spectator_uses_update_display(monkeypatch: pytest.MonkeyPatch) -> None:
    ui = build_live_spectator_widgets()
    close_all = MagicMock()
    clear = MagicMock()
    update = MagicMock()
    monkeypatch.setattr("agent_rpg.live_spectator_ui.W.Widget.close_all", close_all)
    monkeypatch.setattr("agent_rpg.live_spectator_ui.clear_output", clear)
    monkeypatch.setattr("agent_rpg.live_spectator_ui.update_display", update)

    display_live_spectator(ui)

    close_all.assert_called_once()
    clear.assert_called_once_with(wait=True)
    update.assert_called_once_with(ui.root, display_id=DISPLAY_ID)


def test_display_live_spectator_falls_back_to_display(monkeypatch: pytest.MonkeyPatch) -> None:
    ui = build_live_spectator_widgets()
    fallback = MagicMock()
    monkeypatch.setattr("agent_rpg.live_spectator_ui.W.Widget.close_all", MagicMock())
    monkeypatch.setattr("agent_rpg.live_spectator_ui.clear_output", MagicMock())
    monkeypatch.setattr(
        "agent_rpg.live_spectator_ui.update_display",
        MagicMock(side_effect=RuntimeError("no comm")),
    )
    monkeypatch.setattr("agent_rpg.live_spectator_ui.display", fallback)

    display_live_spectator(ui)

    fallback.assert_called_once_with(ui.root, display_id=DISPLAY_ID)


def test_ensure_ipywidgets_raises_without_package(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ipywidgets":
            raise ImportError("no ipywidgets")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="ipywidgets is required"):
        ensure_ipywidgets()
