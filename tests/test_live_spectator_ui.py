"""Live spectator widget factory (notebook 08 controls)."""

from agent_rpg.live_spectator_ui import _short, build_live_spectator_widgets
from agent_rpg.model_catalog import HF_ROUTER_INSTRUCT_MODEL_IDS, hf_router_model_entries


def _local_only_label(ui) -> str:
    for label in ui.labels:
        if ui.label_to_id[label] not in HF_ROUTER_INSTRUCT_MODEL_IDS:
            return label
    raise AssertionError("expected at least one non-HF-router catalog label")


def _hf_router_label(ui) -> str:
    hf_labels = [m["label"] for m in hf_router_model_entries()]
    assert hf_labels, "expected HF router entries in catalog"
    return hf_labels[0]


def test_build_live_spectator_widgets_has_catalog_labels():
    ui = build_live_spectator_widgets()
    assert len(ui.labels) >= 1
    assert ui.labels[0] in ui.label_to_id
    assert ui.exec_mode.value in ("HF Inference API", "Local Transformers")
    assert len(ui.agent_dds) == 8
    assert ui.root.children and len(ui.root.children) == 4


def test_short_truncates_long_goal_labels():
    assert _short("x" * 52) == "x" * 52
    assert _short("x" * 53) == "x" * 51 + "…"


def test_hf_pool_guard_strips_non_router_models():
    """HF Inference mode must not keep 1.5B/3B pool picks (model_not_supported risk)."""
    ui = build_live_spectator_widgets()
    local_label = _local_only_label(ui)
    hf_label = _hf_router_label(ui)

    ui.pool_sel.value = (local_label, hf_label)
    assert ui.pool_sel.value == (hf_label,)


def test_hf_pool_guard_resets_to_default_when_all_invalid():
    ui = build_live_spectator_widgets()
    local_label = _local_only_label(ui)
    hf_label = _hf_router_label(ui)

    ui.pool_sel.value = (local_label,)
    assert ui.pool_sel.value == (hf_label,)


def test_hf_pool_guard_ignored_in_local_transformers_mode():
    ui = build_live_spectator_widgets()
    local_label = _local_only_label(ui)

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


def test_default_pool_prefers_hf_router_model():
    ui = build_live_spectator_widgets()
    hf_label = _hf_router_label(ui)
    assert ui.pool_sel.value == (hf_label,)
