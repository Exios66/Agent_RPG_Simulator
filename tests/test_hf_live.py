import os

import pytest

from agent_rpg.backends.hf_inference import HuggingFaceInferenceBackend


@pytest.mark.live_hf
def test_hf_inference_smoke():
    if os.environ.get("RUN_HF_LIVE") != "1" or not os.environ.get("HF_TOKEN"):
        pytest.skip("Set RUN_HF_LIVE=1 and HF_TOKEN for live HF test")
    b = HuggingFaceInferenceBackend()
    out = b.generate(
        [{"role": "user", "content": 'Reply with JSON: {"thought":"","say":"pong","directed_at":null}'}],
        model_id="HuggingFaceH4/zephyr-7b-beta",
        max_new_tokens=64,
        temperature=0.1,
    )
    assert "pong" in out.lower() or "say" in out.lower()
