"""Hugging Face Inference backend HTTP error hints."""

from __future__ import annotations

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

from agent_rpg.backends.hf_inference import _reraise_inference_http_error


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


def test_other_status_reraises_unchained_message() -> None:
    err = _err(500, "server error")
    with pytest.raises(HfHubHTTPError) as info:
        _reraise_inference_http_error(err)
    assert info.value is err
    assert "[agent-rpg]" not in str(info.value)
