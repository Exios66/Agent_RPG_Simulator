from __future__ import annotations

import os
from typing import Any

from huggingface_hub import InferenceClient


class HuggingFaceInferenceBackend:
    """Remote inference via Hugging Face (InferenceClient.chat_completion)."""

    def __init__(self, token: str | None = None, base_url: str | None = None) -> None:
        self._token = token or os.environ.get("HF_TOKEN")
        self._base_url = base_url

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
        client_kwargs: dict[str, Any] = {}
        if self._token:
            client_kwargs["token"] = self._token
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        client = InferenceClient(**client_kwargs)
        extra: dict[str, Any] = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            extra["top_p"] = top_p
        extra.update(kwargs)
        completion = client.chat_completion(messages=messages, model=model_id, **extra)
        choice = completion.choices[0]
        msg = choice.message
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
                else:
                    parts.append(str(block))
            return "".join(parts)
        return str(content or "")
