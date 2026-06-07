from agent_rpg.model_catalog import (
    DEFAULT_HF_INFERENCE_MODEL_ID,
    DEFAULT_INSTRUCT_MODEL_ID,
    HF_ROUTER_INSTRUCT_MODEL_IDS,
    SMALL_INSTRUCT_MODELS,
    default_model_id_for_execution,
    hf_router_model_entries,
    label_to_id_map,
    labels_by_id,
    model_ids_for_widgets,
    warn_if_not_hf_router,
)


def test_hf_and_local_defaults_differ() -> None:
    assert DEFAULT_HF_INFERENCE_MODEL_ID != DEFAULT_INSTRUCT_MODEL_ID
    assert DEFAULT_HF_INFERENCE_MODEL_ID in HF_ROUTER_INSTRUCT_MODEL_IDS
    assert DEFAULT_INSTRUCT_MODEL_ID not in HF_ROUTER_INSTRUCT_MODEL_IDS


def test_default_model_id_for_execution() -> None:
    assert default_model_id_for_execution("HF Inference API") == DEFAULT_HF_INFERENCE_MODEL_ID
    assert default_model_id_for_execution("Local Transformers") == DEFAULT_INSTRUCT_MODEL_ID


def test_hf_router_entries_are_catalog_subset() -> None:
    ids = {m["id"] for m in hf_router_model_entries()}
    assert ids == set(HF_ROUTER_INSTRUCT_MODEL_IDS)


def test_warn_if_not_hf_router() -> None:
    bad = warn_if_not_hf_router(
        [DEFAULT_INSTRUCT_MODEL_ID, DEFAULT_HF_INFERENCE_MODEL_ID],
        execution_mode="HF Inference API",
    )
    assert bad == [DEFAULT_INSTRUCT_MODEL_ID]
    assert warn_if_not_hf_router([DEFAULT_INSTRUCT_MODEL_ID], execution_mode="local") == []


def test_widget_label_maps_match_catalog() -> None:
    l2i = label_to_id_map()
    i2l = labels_by_id()
    widget_pairs = model_ids_for_widgets()
    assert l2i == {m["label"]: m["id"] for m in SMALL_INSTRUCT_MODELS}
    assert i2l == {m["id"]: m["label"] for m in SMALL_INSTRUCT_MODELS}
    assert widget_pairs == [(m["label"], m["id"]) for m in SMALL_INSTRUCT_MODELS]
    assert set(l2i.values()) == {m["id"] for m in SMALL_INSTRUCT_MODELS}


def test_default_model_id_for_execution_aliases() -> None:
    assert default_model_id_for_execution("hf") == DEFAULT_HF_INFERENCE_MODEL_ID
    assert default_model_id_for_execution("  HF_INFERENCE  ") == DEFAULT_HF_INFERENCE_MODEL_ID
    assert default_model_id_for_execution("openrouter") == DEFAULT_INSTRUCT_MODEL_ID
