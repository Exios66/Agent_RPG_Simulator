from __future__ import annotations

import json
import os
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _content_from_choice(choice: dict[str, Any]) -> str:
    # Some proxies or malformed bodies use a scalar ``message``; only objects have ``.get``.
    msg_raw = choice.get("message")
    msg = msg_raw if isinstance(msg_raw, dict) else {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content or "")


class OpenRouterBackend:
    """Remote chat completions via the OpenRouter API (OpenAI-compatible ``/chat/completions``).

    Reads ``OPENROUTER_API_KEY`` from the environment when ``api_key`` is omitted.
    Optional: ``OPENROUTER_BASE_URL`` (default ``https://openrouter.ai/api/v1``),
    ``OPENROUTER_HTTP_REFERER``, ``OPENROUTER_APP_TITLE`` for OpenRouter rankings headers.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        http_referer: str | None = None,
        app_title: str | None = None,
    ) -> None:
        self._api_key = os.environ.get("OPENROUTER_API_KEY") if api_key is None else api_key
        self._base_url = (base_url or os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE).rstrip("/")
        self._http_referer = http_referer or os.environ.get("OPENROUTER_HTTP_REFERER")
        self._app_title = app_title or os.environ.get("OPENROUTER_APP_TITLE")

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model_id: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float | None = None,
        **kwargs: Any,
    ) -> str:
        if not self._api_key:
            raise ValueError(
                "OpenRouterBackend requires an API key: pass api_key=... or set OPENROUTER_API_KEY in the environment."
            )

        chunk_callback: Callable[[str], None] | None = kwargs.pop("chunk_callback", None)
        stream = bool(kwargs.pop("stream", False))

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._http_referer:
            headers["HTTP-Referer"] = self._http_referer
        if self._app_title:
            headers["X-Title"] = self._app_title

        body: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            body["top_p"] = top_p
        if stream:
            body["stream"] = True

        passthrough = (
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "stop",
            "response_format",
            "user",
        )
        for key in passthrough:
            if key in kwargs:
                body[key] = kwargs[key]

        data = json.dumps(body).encode("utf-8")
        req = Request(url, data=data, headers=headers, method="POST")

        try:
            resp = urlopen(req, timeout=300)
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(
                f"OpenRouter HTTP {e.code}: {detail or e.reason}. "
                "Check OPENROUTER_API_KEY, model_id (e.g. a :free slug from openrouter.ai/models), and quotas."
            ) from e
        except URLError as e:
            raise RuntimeError(f"OpenRouter request failed: {e.reason!r}") from e

        if stream:
            return self._read_sse_stream(resp, chunk_callback)
        try:
            raw_text = resp.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"OpenRouter returned invalid JSON ({e}). Body prefix: {raw_text[:500]!r}"
                ) from e
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"OpenRouter returned JSON that is not an object: {type(payload).__name__!r} {payload!r}"
                )
        finally:
            resp.close()
        if not isinstance(payload, dict):
            raise RuntimeError(f"OpenRouter returned non-object JSON: {payload!r}")
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter returned no choices: {payload!r}")
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError(
                f"OpenRouter returned invalid choice (expected object, got {type(first).__name__!r}): {payload!r}"
            )
        return _content_from_choice(first)

    def _read_sse_stream(self, resp: Any, chunk_callback: Callable[[str], None] | None) -> str:
        parts: list[str] = []
        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("error") is not None:
                    err = obj["error"]
                    if isinstance(err, dict):
                        msg = err.get("message") or err.get("code") or json.dumps(err)
                    else:
                        msg = str(err)
                    raise RuntimeError(f"OpenRouter stream error: {msg}") from None
                for choice in obj.get("choices") or []:
                    if not isinstance(choice, dict):
                        continue
                    delta_raw = choice.get("delta")
                    delta = delta_raw if isinstance(delta_raw, dict) else {}
                    piece = delta.get("content")
                    if isinstance(piece, str) and piece:
                        parts.append(piece)
                        if chunk_callback is not None:
                            chunk_callback(piece)
        finally:
            resp.close()
        return "".join(parts)
