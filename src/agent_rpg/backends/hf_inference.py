from __future__ import annotations

import os
from typing import Any, Callable

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError


def _normalize_message_content(content: Any) -> str:
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


def _reraise_inference_http_error(exc: HfHubHTTPError) -> None:
    """Add short operator hints for common Inference Provider / router failures."""
    resp = exc.response
    code = resp.status_code
    if code == 402:
        hint = (
            "\n\n[agent-rpg] HTTP 402: Hugging Face Inference billing or included credits exhausted. "
            "See https://huggingface.co/settings/billing (credits / PRO). "
            "For local runs without the router: use `FakeLLMBackend`, or "
            "`pip install -e '.[local]'` and `TransformersLocalBackend`, or fewer rounds/agents."
        )
        raise HfHubHTTPError(
            str(exc) + hint,
            response=resp,
            server_message=getattr(exc, "server_message", None),
        ) from exc
    if code == 403:
        hint = (
            "\n\n[agent-rpg] HTTP 403: token may lack access or the model is gated — accept the license on the model card "
            "and ensure `HF_TOKEN` has read access."
        )
        raise HfHubHTTPError(
            str(exc) + hint,
            response=resp,
            server_message=getattr(exc, "server_message", None),
        ) from exc
    if code == 400:
        detail = str(exc).lower()
        if "model_not_supported" in detail or "not supported by any provider" in detail:
            hint = (
                "\n\n[agent-rpg] HTTP 400: this model id is not enabled on the Hugging Face Inference router for your "
                "account. Use `agent_rpg.model_catalog.DEFAULT_HF_INFERENCE_MODEL_ID` (currently Qwen 7B) or another id "
                "in `HF_ROUTER_INSTRUCT_MODEL_IDS`, browse https://huggingface.co/inference/models , or switch to "
                "**Local Transformers** in notebook 08."
            )
            raise HfHubHTTPError(
                str(exc) + hint,
                response=resp,
                server_message=getattr(exc, "server_message", None),
            ) from exc
    raise exc


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
        chunk_callback: Callable[[str], None] | None = kwargs.pop("chunk_callback", None)
        stream = bool(kwargs.pop("stream", False))

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

        if stream:
            extra["stream"] = True
            parts: list[str] = []
            try:
                stream_iter = client.chat_completion(messages=messages, model=model_id, **extra)
                for chunk in stream_iter:
                    choice0 = chunk.choices[0] if getattr(chunk, "choices", None) else None
                    if choice0 is None:
                        continue
                    delta = getattr(choice0, "delta", None)
                    piece: str | None = None
                    if delta is not None:
                        piece = getattr(delta, "content", None)
                    if not piece and getattr(choice0, "message", None) is not None:
                        piece = getattr(choice0.message, "content", None)
                    text = _normalize_message_content(piece) if piece is not None else ""
                    if text:
                        parts.append(text)
                        if chunk_callback is not None:
                            chunk_callback(text)
            except HfHubHTTPError as e:
                _reraise_inference_http_error(e)
            return "".join(parts)

        try:
            completion = client.chat_completion(messages=messages, model=model_id, **extra)
        except HfHubHTTPError as e:
            _reraise_inference_http_error(e)
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise RuntimeError(
                "Hugging Face chat completion returned no choices; check model_id, provider status, and quotas."
            )
        choice = choices[0]
        if choice is None:
            raise RuntimeError(
                "Hugging Face chat completion returned invalid choice (expected object, got None); "
                "check model_id, provider status, and quotas."
            )
        msg = getattr(choice, "message", None)
        content = getattr(msg, "content", None) if msg is not None else None
        return _normalize_message_content(content)
