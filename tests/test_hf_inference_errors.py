"""Hugging Face Inference backend HTTP error hints."""

from __future__ import annotations

import httpx
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import HfHubHTTPError

from agent_rpg.backends.hf_inference import HuggingFaceInferenceBackend, _reraise_inference_http_error


def _err(code: int, msg: str = "upstream") -> HfHubHTTPError:
    req = httpx.Request("POST", "https://router.huggingface.co/v1/chat/completions")
    resp = httpx.Response(code, request=req)
    return HfHubHTTPError(msg, response=resp)


def test_402_wraps_with_billing_hint() -> None:
    with pytest.raises(HfHubHTTPError) as info:
        _reraise_inference_http_error(_err(402, "Payment Required"))
    assert "[agent-rpg]" in str(info.value)
    assert "402" in str(info.value) or "billing" in str(info.value).lower()


def test_403_wraps_with_access_hint() -> None:
    with pytest.raises(HfHubHTTPError) as info:
        _reraise_inference_http_error(_err(403, "Forbidden"))
    assert "[agent-rpg]" in str(info.value)
    assert "403" in str(info.value) or "gated" in str(info.value).lower()


def test_400_model_not_supported_wraps_with_catalog_hint() -> None:
    msg = "model_not_supported The requested model 'x' is not supported by any provider"
    with pytest.raises(HfHubHTTPError) as info:
        _reraise_inference_http_error(_err(400, msg))
    assert "[agent-rpg]" in str(info.value)
    assert "SMALL_INSTRUCT_MODELS" in str(info.value) or "inference/models" in str(info.value)


def test_400_unrelated_reraises_original() -> None:
    err = _err(400, "Bad request: malformed")
    with pytest.raises(HfHubHTTPError) as info:
        _reraise_inference_http_error(err)
    assert info.value is err
    assert "[agent-rpg]" not in str(info.value)


def test_generate_non_stream_empty_choices_raises() -> None:
    """Router must not IndexError when ``choices`` is empty."""
    completion = MagicMock()
    completion.choices = []
    fake_client = MagicMock()
    fake_client.chat_completion.return_value = completion

    with patch("agent_rpg.backends.hf_inference.InferenceClient", return_value=fake_client):
        b = HuggingFaceInferenceBackend(token="hf_test")
        with pytest.raises(RuntimeError, match="no choices"):
            b.generate([{"role": "user", "content": "x"}], model_id="dummy/model")


def test_other_status_reraises_unchained_message() -> None:
    err = _err(500, "server error")
    with pytest.raises(HfHubHTTPError) as info:
        _reraise_inference_http_error(err)
    assert info.value is err
    assert "[agent-rpg]" not in str(info.value)
