import os

import pytest

from agent_rpg.backends.hf_inference import HuggingFaceInferenceBackend
from agent_rpg.model_catalog import DEFAULT_INSTRUCT_MODEL_ID


@pytest.mark.live_hf
def test_hf_inference_smoke():
    if os.environ.get("RUN_HF_LIVE") != "1" or not os.environ.get("HF_TOKEN"):
        pytest.skip("Set RUN_HF_LIVE=1 and HF_TOKEN for live HF test")
    model_id = os.environ.get("HF_LIVE_MODEL", DEFAULT_INSTRUCT_MODEL_ID)
    b = HuggingFaceInferenceBackend()
    out = b.generate(
        [{"role": "user", "content": 'Reply with JSON: {"thought":"","say":"pong","directed_at":null}'}],
        model_id=model_id,
        max_new_tokens=64,
        temperature=0.1,
    )
    assert "pong" in out.lower() or "say" in out.lower()
