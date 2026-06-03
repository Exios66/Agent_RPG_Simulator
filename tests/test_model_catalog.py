from agent_rpg.model_catalog import (
    DEFAULT_HF_INFERENCE_MODEL_ID,
    DEFAULT_INSTRUCT_MODEL_ID,
    HF_ROUTER_INSTRUCT_MODEL_IDS,
    default_model_id_for_execution,
    hf_router_model_entries,
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
