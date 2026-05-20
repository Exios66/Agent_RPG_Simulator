"""Unit tests for ``OpenRouterBackend`` (HTTP mocked; no network)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent_rpg.backends.openrouter import OpenRouterBackend


def test_generate_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    b = OpenRouterBackend(api_key=None)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        b.generate([{"role": "user", "content": "x"}], model_id="m")


def test_generate_non_stream_rejects_null_choice() -> None:
    """``choices: [null]`` must not crash with AttributeError (invalid/mangled JSON)."""
    payload = {"choices": [None]}
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode()

    with patch("agent_rpg.backends.openrouter.urlopen", return_value=mock_resp):
        b = OpenRouterBackend(api_key="sk-or-test")
        with pytest.raises(RuntimeError, match="invalid choice"):
            b.generate([{"role": "user", "content": "ping"}], model_id="m")


def test_generate_non_stream_parses_message_content() -> None:
    payload = {"choices": [{"message": {"content": '{"say":"hi"}'}}]}
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode()

    with patch("agent_rpg.backends.openrouter.urlopen", return_value=mock_resp) as m_url:
        b = OpenRouterBackend(api_key="sk-or-test")
        out = b.generate([{"role": "user", "content": "ping"}], model_id="org/model:free")

    assert out == '{"say":"hi"}'
    m_url.assert_called_once()
    req = m_url.call_args[0][0]
    assert req.full_url.endswith("/chat/completions")
    body = json.loads(req.data.decode())
    assert body["model"] == "org/model:free"
    assert body["messages"][0]["role"] == "user"


def test_generate_non_stream_invalid_json_raises_clear_runtime_error() -> None:
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"not json at all"

    with patch("agent_rpg.backends.openrouter.urlopen", return_value=mock_resp):
        b = OpenRouterBackend(api_key="sk-or-test")
        with pytest.raises(RuntimeError, match="invalid JSON"):
            b.generate([{"role": "user", "content": "x"}], model_id="m")


def test_generate_non_stream_json_array_raises_clear_runtime_error() -> None:
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"[]"

    with patch("agent_rpg.backends.openrouter.urlopen", return_value=mock_resp):
        b = OpenRouterBackend(api_key="sk-or-test")
        with pytest.raises(RuntimeError, match="not an object"):
            b.generate([{"role": "user", "content": "x"}], model_id="m")


def test_generate_non_stream_empty_choices_raises() -> None:
    payload = {"choices": []}
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode()

    with patch("agent_rpg.backends.openrouter.urlopen", return_value=mock_resp):
        b = OpenRouterBackend(api_key="sk-or-test")
        with pytest.raises(RuntimeError, match="no choices"):
            b.generate([{"role": "user", "content": "x"}], model_id="m")


def test_generate_non_stream_list_message_content() -> None:
    """Multimodal-style content blocks must concatenate to a single string."""
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"text": "world"},
                    ]
                }
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode()

    with patch("agent_rpg.backends.openrouter.urlopen", return_value=mock_resp):
        b = OpenRouterBackend(api_key="sk-or-test")
        out = b.generate([{"role": "user", "content": "ping"}], model_id="m")

    assert out == "Hello world"


def test_generate_non_stream_scalar_message_does_not_crash() -> None:
    """A non-object ``message`` must not cause AttributeError on ``.get``."""
    payload = {"choices": [{"message": "upstream error string"}]}
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode()

    with patch("agent_rpg.backends.openrouter.urlopen", return_value=mock_resp):
        b = OpenRouterBackend(api_key="sk-or-test")
        out = b.generate([{"role": "user", "content": "ping"}], model_id="m")

    assert out == ""


def test_generate_stream_raises_on_error_event() -> None:
    """Streaming must not swallow top-level ``error`` SSE payloads (OpenAI-compatible)."""
    lines = [
        b'data: {"error":{"message":"rate limit","type":"requests"}}\n',
        b"data: [DONE]\n",
    ]

    class FakeStream:
        def __init__(self, data: list[bytes]) -> None:
            self._data = data

        def __iter__(self):
            return iter(self._data)

        def close(self) -> None:
            pass

    stream = FakeStream(lines)
    with patch("agent_rpg.backends.openrouter.urlopen", return_value=stream):
        b = OpenRouterBackend(api_key="k")
        with pytest.raises(RuntimeError, match="OpenRouter stream error"):
            b.generate([{"role": "user", "content": "x"}], model_id="m", stream=True)


def test_generate_stream_skips_non_dict_choice_entries() -> None:
    """SSE lines with null or non-object choices must not crash streaming."""
    lines = [
        b'data: {"choices":[null]}\n\n',
        b'data: {"choices":["not-a-dict"]}\n\n',
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
        b"data: [DONE]\n",
    ]

    class FakeStream:
        def __init__(self, data: list[bytes]) -> None:
            self._data = data

        def __iter__(self):
            return iter(self._data)

        def close(self) -> None:
            pass

    stream = FakeStream(lines)
    with patch("agent_rpg.backends.openrouter.urlopen", return_value=stream):
        b = OpenRouterBackend(api_key="k")
        out = b.generate(
            [{"role": "user", "content": "x"}],
            model_id="m",
            stream=True,
        )

    assert out == "ok"


def test_generate_stream_non_object_delta_skipped_without_crash() -> None:
    lines = [
        b'data: {"choices":[{"delta":"broken"}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
        b"data: [DONE]\n",
    ]

    class FakeStream:
        def __init__(self, data: list[bytes]) -> None:
            self._data = data

        def __iter__(self):
            return iter(self._data)

        def close(self) -> None:
            pass

    stream = FakeStream(lines)
    with patch("agent_rpg.backends.openrouter.urlopen", return_value=stream):
        b = OpenRouterBackend(api_key="k")
        out = b.generate([{"role": "user", "content": "x"}], model_id="m", stream=True)

    assert out == "ok"


def test_generate_stream_accumulates_delta() -> None:
    lines = [
        b'data: {"choices":[{"delta":{"content":"He"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n',
        b"data: [DONE]\n",
    ]

    class FakeStream:
        def __init__(self, data: list[bytes]) -> None:
            self._data = data

        def __iter__(self):
            return iter(self._data)

        def read(self, n: int = -1) -> bytes:
            raise AssertionError("streaming path must not call read()")

        def close(self) -> None:
            pass

    stream = FakeStream(lines)
    with patch("agent_rpg.backends.openrouter.urlopen", return_value=stream):
        b = OpenRouterBackend(api_key="k")
        chunks: list[str] = []

        def cb(s: str) -> None:
            chunks.append(s)

        out = b.generate(
            [{"role": "user", "content": "x"}],
            model_id="m",
            stream=True,
            chunk_callback=cb,
        )

    assert out == "Hello"
    assert chunks == ["He", "llo"]
