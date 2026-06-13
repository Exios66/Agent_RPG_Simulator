"""Live spectator widget factory (notebook 08 controls)."""

from __future__ import annotations

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
    assert _short("short") == "short"
    assert len(_short("x" * 60)) == 52
    assert _short("x" * 60).endswith("…")


def test_exec_mode_toggles_stream_and_device():
    ui = build_live_spectator_widgets()
    assert ui.exec_mode.value == "HF Inference API"
    assert ui.stream_tokens.disabled is False
    assert ui.device_map.disabled is True

    ui.exec_mode.value = "Local Transformers"
    assert ui.stream_tokens.disabled is True
    assert ui.device_map.disabled is False


def test_agent_rows_show_first_n_dropdowns():
    ui = build_live_spectator_widgets()
    ui.assign_mode.value = "Pick per agent"

    ui.n_agents.value = 3
    for i, dd in enumerate(ui.agent_dds):
        expected = None if i < 3 else "none"
        assert dd.layout.display == expected, f"agent {i} at n=3"

    ui.n_agents.value = 5
    for i, dd in enumerate(ui.agent_dds):
        expected = None if i < 5 else "none"
        assert dd.layout.display == expected, f"agent {i} at n=5"


def test_hf_pool_guard_rejects_non_router_models():
    ui = build_live_spectator_widgets()
    hf_default = ui.pool_sel.value[0]
    assert ui.label_to_id[hf_default] in HF_ROUTER_INSTRUCT_MODEL_IDS

    non_router = next(
        label for label in ui.labels if ui.label_to_id[label] not in HF_ROUTER_INSTRUCT_MODEL_IDS
    )
    ui.pool_sel.value = (non_router,)
    assert ui.pool_sel.value == (hf_default,)

    router = next(
        label for label in ui.labels if ui.label_to_id[label] in HF_ROUTER_INSTRUCT_MODEL_IDS
    )
    ui.pool_sel.value = (non_router, router)
    assert ui.pool_sel.value == (router,)


def test_hf_pool_guard_skipped_in_local_mode():
    ui = build_live_spectator_widgets()
    non_router = next(
        label for label in ui.labels if ui.label_to_id[label] not in HF_ROUTER_INSTRUCT_MODEL_IDS
    )
    ui.exec_mode.value = "Local Transformers"
    ui.pool_sel.value = (non_router,)
    assert ui.pool_sel.value == (non_router,)
