from __future__ import annotations

from typing import Any

from agent_rpg.backends.base import LLMBackend


class TransformersLocalBackend:
    """Local inference via transformers pipeline (optional dependency)."""

    def __init__(self, device_map: str | dict[str, Any] | None = None) -> None:
        self._device_map = device_map or "auto"
        self._pipelines: dict[str, Any] = {}

    def _get_pipeline(self, model_id: str) -> Any:
        if model_id in self._pipelines:
            return self._pipelines[model_id]
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError as e:
            raise ImportError(
                "Install optional deps: pip install 'agent-rpg[local]'"
            ) from e

        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=self._device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            trust_remote_code=True,
        )
        self._pipelines[model_id] = pipe
        return pipe

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
        pipe = self._get_pipeline(model_id)
        tok = pipe.tokenizer
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            **kwargs,
        }
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        out = pipe(prompt, **gen_kwargs)
        if not out:
            return ""
        generated = out[0].get("generated_text", "")
        if generated.startswith(prompt):
            return generated[len(prompt) :].strip()
        return generated.strip()
