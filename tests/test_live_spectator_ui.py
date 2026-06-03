"""Live spectator widget factory (notebook 08 controls)."""

from agent_rpg.live_spectator_ui import _short, build_live_spectator_widgets
from agent_rpg.model_catalog import (
    HF_ROUTER_INSTRUCT_MODEL_IDS,
    SMALL_INSTRUCT_MODELS,
    hf_router_model_entries,
)


def test_build_live_spectator_widgets_has_catalog_labels():
    ui = build_live_spectator_widgets()
    assert len(ui.labels) >= 1
    assert ui.labels[0] in ui.label_to_id
    assert ui.exec_mode.value in ("HF Inference API", "Local Transformers")
    assert len(ui.agent_dds) == 8
    assert ui.root.children and len(ui.root.children) == 4


def test_short_truncates_long_strings() -> None:
    assert _short("abc", 10) == "abc"
    assert _short("x" * 60, 52).endswith("…")
    assert len(_short("x" * 60, 52)) == 52


def test_hf_pool_guard_resets_non_router_selection() -> None:
    """Notebook 08 must not leave 1.5B/3B in the HF pool (model_not_supported)."""
    ui = build_live_spectator_widgets()
    local_label = next(
        m["label"] for m in SMALL_INSTRUCT_MODELS if m["id"] not in HF_ROUTER_INSTRUCT_MODEL_IDS
    )
    hf_default = hf_router_model_entries()[0]["label"]

    ui.exec_mode.value = "HF Inference API"
    ui.pool_sel.value = (local_label,)

    assert ui.pool_sel.value == (hf_default,)


def test_hf_pool_guard_keeps_only_router_models_when_mixed() -> None:
    ui = build_live_spectator_widgets()
    router_labels = [m["label"] for m in hf_router_model_entries()]
    local_label = next(
        m["label"] for m in SMALL_INSTRUCT_MODELS if m["id"] not in HF_ROUTER_INSTRUCT_MODEL_IDS
    )
    assert len(router_labels) >= 1

    ui.exec_mode.value = "HF Inference API"
    ui.pool_sel.value = (router_labels[0], local_label)

    assert ui.pool_sel.value == (router_labels[0],)


def test_hf_pool_guard_noop_in_local_mode() -> None:
    ui = build_live_spectator_widgets()
    local_label = next(
        m["label"] for m in SMALL_INSTRUCT_MODELS if m["id"] not in HF_ROUTER_INSTRUCT_MODEL_IDS
    )

    ui.exec_mode.value = "Local Transformers"
    ui.pool_sel.value = (local_label,)

    assert ui.pool_sel.value == (local_label,)
