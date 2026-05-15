from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal contract for chat-style generation."""

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
        """Return assistant text (raw string; caller parses structured JSON if needed)."""
        ...
