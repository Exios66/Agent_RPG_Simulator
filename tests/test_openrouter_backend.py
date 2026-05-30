"""Unit tests for ``OpenRouterBackend`` (HTTP mocked; no network)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent_rpg.backends.openrouter import OpenRouterBackend, _content_from_choice
from urllib.error import HTTPError, URLError


def test_generate_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    b = OpenRouterBackend(api_key=None)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        b.generate([{"role": "user", "content": "x"}], model_id="m")


def test_generate_non_stream_empty_choices_raises() -> None:
    payload: dict = {"choices": []}
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode()

    with patch("agent_rpg.backends.openrouter.urlopen", return_value=mock_resp):
        b = OpenRouterBackend(api_key="sk-or-test")
        with pytest.raises(RuntimeError, match="no choices"):
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


def test_generate_stream_skips_non_dict_choice_entries() -> None:
    """SSE chunks with scalar/null choices must not crash (regression for #6)."""
    lines = [
        b'data: {"choices":[null,{"delta":{"content":"ok"}}]}\n\n',
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
        out = b.generate(
            [{"role": "user", "content": "x"}],
            model_id="m",
            stream=True,
        )

    assert out == "ok"


def test_content_from_choice_list_blocks_with_text_key() -> None:
    choice = {
        "message": {
            "content": [
                {"text": "Hello "},
                {"type": "text", "text": "world"},
            ]
        }
    }
    assert _content_from_choice(choice) == "Hello world"


def test_content_from_choice_list_unknown_block_str() -> None:
    choice = {"message": {"content": [42, {"text": "x"}]}}
    assert _content_from_choice(choice) == "42x"


def test_generate_non_stream_multipart_message_content() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"text": "part1"},
                        {"type": "text", "text": "part2"},
                    ]
                }
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode()

    with patch("agent_rpg.backends.openrouter.urlopen", return_value=mock_resp):
        b = OpenRouterBackend(api_key="sk-or-test")
        out = b.generate([{"role": "user", "content": "x"}], model_id="m")

    assert out == "part1part2"


def test_generate_http_error_includes_status_and_body_prefix() -> None:
    err_body = b'{"error":"quota"}'
    http_err = HTTPError(
        url="https://openrouter.ai/api/v1/chat/completions",
        code=429,
        msg="Too Many Requests",
        hdrs=None,
        fp=MagicMock(read=MagicMock(return_value=err_body)),
    )

    with patch("agent_rpg.backends.openrouter.urlopen", side_effect=http_err):
        b = OpenRouterBackend(api_key="k")
        with pytest.raises(RuntimeError, match="HTTP 429"):
            b.generate([{"role": "user", "content": "x"}], model_id="m")


def test_generate_url_error_wraps_reason() -> None:
    with patch(
        "agent_rpg.backends.openrouter.urlopen",
        side_effect=URLError("connection reset"),
    ):
        b = OpenRouterBackend(api_key="k")
        with pytest.raises(RuntimeError, match="request failed"):
            b.generate([{"role": "user", "content": "x"}], model_id="m")
