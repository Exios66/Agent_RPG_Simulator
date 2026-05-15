from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any


class FakeLLMBackend:
    """Test double: returns queued strings or uses a factory keyed by call index."""

    def __init__(
        self,
        responses: list[str] | None = None,
        factory: Callable[[int, list[dict[str, str]]], str] | None = None,
    ) -> None:
        self._queue: Iterator[str] | None = iter(responses) if responses else None
        self._factory = factory
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []
        self._index = 0

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
        opts: dict[str, Any] = {
            "model_id": model_id,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs,
        }
        self.calls.append((messages, opts))
        if self._factory is not None:
            out = self._factory(self._index, messages)
            self._index += 1
            return out
        if self._queue is not None:
            try:
                nxt = next(self._queue)
                self._index += 1
                return nxt
            except StopIteration as e:
                raise RuntimeError("FakeLLMBackend: no more queued responses") from e
        return '{"thought":"","say":"...","directed_at":null}'
